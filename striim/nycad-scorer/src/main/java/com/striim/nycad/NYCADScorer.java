package com.striim.nycad;

import com.webaction.anno.AdapterType;
import com.webaction.anno.PropertyTemplate;
import com.webaction.anno.PropertyTemplateProperty;
import com.webaction.runtime.components.openprocessor.StriimOpenProcessor;
import com.webaction.runtime.containers.WAEvent;
import com.webaction.runtime.containers.IBatch;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

@PropertyTemplate(
    name = "NYCADScorer",
    type = AdapterType.process,
    properties = {
        @PropertyTemplateProperty(name = "apiEndpoint", type = String.class,
            required = false, defaultValue = "http://localhost:8000/v1/score"),
        @PropertyTemplateProperty(name = "timeoutMs", type = Integer.class,
            required = false, defaultValue = "5000"),
        @PropertyTemplateProperty(name = "maxRetries", type = Integer.class,
            required = false, defaultValue = "3")
    },
    outputType = com.webaction.proc.events.WAEvent.class,
    inputType  = com.webaction.proc.events.WAEvent.class
)
public class NYCADScorer extends StriimOpenProcessor {

    private static final Logger logger = LogManager.getLogger(NYCADScorer.class);
    private static final int WINDOW_SIZE = 336;

    private HttpClient httpClient;
    private String apiEndpoint;
    private int timeoutMs;
    private int maxRetries;

    // Internal sliding window buffer
    private final ArrayList<String> valueBuffer = new ArrayList<>(400);
    private final ArrayList<String> timestampBuffer = new ArrayList<>(400);

    // Counters for observability
    private long rowsReceived = 0;
    private long windowsScored = 0;
    private long windowsSkipped = 0;
    private long anomaliesDetected = 0;

    @Override
    public void run() {
        if (httpClient == null) {
            int effectiveTimeout = (timeoutMs > 0) ? timeoutMs : 5000;
            int effectiveRetries = (maxRetries > 0) ? maxRetries : 3;
            this.timeoutMs = effectiveTimeout;
            this.maxRetries = effectiveRetries;
            httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(effectiveTimeout))
                .build();
            if (apiEndpoint == null || apiEndpoint.isEmpty()) {
                apiEndpoint = "http://localhost:8000/v1/score";
            }
            logger.info("NYCADScorer initialized: endpoint={}, timeout={}ms, retries={}",
                apiEndpoint, effectiveTimeout, effectiveRetries);
        }

        IBatch<WAEvent> batch = getAdded();
        if (batch == null) return;

        Iterator<WAEvent> it = batch.iterator();
        while (it.hasNext()) {
            WAEvent outerEvent = it.next();

            try {
                // Extract raw CSV fields from the inner WAEvent
                Object innerObj = outerEvent.data;
                Object[] fields;

                if (innerObj != null &&
                    innerObj.getClass().getName().contains("WAEvent")) {
                    java.lang.reflect.Field dataField = innerObj.getClass().getField("data");
                    fields = (Object[]) dataField.get(innerObj);
                } else if (innerObj instanceof Object[]) {
                    fields = (Object[]) innerObj;
                } else {
                    logger.warn("Unexpected event type: {}",
                        innerObj != null ? innerObj.getClass().getName() : "null");
                    continue;
                }

                // DSVParser output: data[0] = timestamp, data[1] = value
                String timestamp = String.valueOf(fields[0]).trim();
                String value = String.valueOf(fields[1]).trim();

                // Add to buffer
                valueBuffer.add(value);
                timestampBuffer.add(timestamp);
                rowsReceived++;

                if (rowsReceived % 1000 == 0) {
                    logger.info("Progress: rows={}, scored={}, skipped={}, anomalies={}",
                        rowsReceived, windowsScored, windowsSkipped, anomaliesDetected);
                }

                // Score when buffer reaches window size
                if (valueBuffer.size() >= WINDOW_SIZE) {
                    String windowStart = timestampBuffer.get(0);
                    String windowEnd = timestampBuffer.get(WINDOW_SIZE - 1);

                    // Build JSON array from buffer
                    StringBuilder valuesJson = new StringBuilder("[");
                    for (int i = 0; i < WINDOW_SIZE; i++) {
                        if (i > 0) valuesJson.append(",");
                        valuesJson.append(valueBuffer.get(i));
                    }
                    valuesJson.append("]");

                    JsonObject body = new JsonObject();
                    body.add("values", JsonParser.parseString(valuesJson.toString()));
                    body.addProperty("window_start", windowStart);
                    body.addProperty("window_end", windowEnd);

                    String responseBody = callApiWithRetry(body.toString());

                    if (responseBody == null) {
                        logger.error("All retries exhausted for window=[{}, {}]",
                            windowStart, windowEnd);
                    } else if (responseBody.isEmpty()) {
                        // 204 No Content -- non-Sunday or training-era window
                        windowsSkipped++;
                    } else {
                        // 200 -- parse and emit result
                        JsonObject resp = JsonParser.parseString(responseBody).getAsJsonObject();

                        String isAnomaly    = resp.get("is_anomaly").getAsBoolean() ? "true" : "false";
                        String anomalyScore = String.valueOf(resp.get("anomaly_score").getAsDouble());
                        String threshold    = String.valueOf(resp.get("threshold").getAsDouble());

                        if ("true".equals(isAnomaly)) anomaliesDetected++;
                        windowsScored++;

                        logger.info("Scored window=[{}, {}] anomaly={} score={}",
                            windowStart, windowEnd, isAnomaly, anomalyScore);

                        // Emit result -- try in-place modification first
                        if (innerObj != null &&
                            innerObj.getClass().getName().contains("WAEvent")) {
                            java.lang.reflect.Field dataField = innerObj.getClass().getField("data");
                            dataField.set(innerObj, new Object[] {
                                isAnomaly, anomalyScore, threshold, windowStart, windowEnd
                            });
                            send(innerObj);
                        } else {
                            send(new Object[] {
                                isAnomaly, anomalyScore, threshold, windowStart, windowEnd
                            });
                        }
                    }

                    // Slide window by 1
                    valueBuffer.remove(0);
                    timestampBuffer.remove(0);
                }

            } catch (Exception e) {
                logger.error("Error processing event: {}", e.getMessage(), e);
            }
        }
    }

    private String callApiWithRetry(String jsonBody) {
        int attempt = 0;
        long backoffMs = 500;
        while (attempt <= maxRetries) {
            try {
                HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(apiEndpoint))
                    .timeout(Duration.ofMillis(timeoutMs))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();
                HttpResponse<String> response = httpClient.send(
                    request, HttpResponse.BodyHandlers.ofString());
                int status = response.statusCode();
                if (status == 204) return "";
                if (status >= 200 && status < 300) return response.body();
                if (status >= 400 && status < 500) {
                    logger.error("Client error: status={}, body={}", status, response.body());
                    return null;
                }
                logger.warn("Server error: status={}, attempt={}/{}", status, attempt, maxRetries);
            } catch (Exception e) {
                logger.warn("API exception: {}, attempt={}/{}", e.getMessage(), attempt, maxRetries);
            }
            try { Thread.sleep(backoffMs); } catch (InterruptedException ie) {
                Thread.currentThread().interrupt(); return null;
            }
            backoffMs *= 2;
            attempt++;
        }
        return null;
    }

    private String normalizeValuesList(String raw) {
        if (raw == null || raw.isEmpty()) return "[]";
        String cleaned = raw.trim();
        if (cleaned.startsWith("[")) cleaned = cleaned.substring(1);
        if (cleaned.endsWith("]"))   cleaned = cleaned.substring(0, cleaned.length() - 1);
        String[] parts = cleaned.split(",");
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < parts.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(parts[i].trim());
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    public void close() throws Exception {
        logger.info("NYCADScorer shutting down. Final: rows={}, scored={}, skipped={}, anomalies={}",
            rowsReceived, windowsScored, windowsSkipped, anomaliesDetected);
    }

    @Override
    public Map getAggVec() { return null; }

    @Override
    public void setAggVec(Map aggVec) { }
}
