package com.striim.lstmae;

import com.webaction.anno.AdapterType;
import com.webaction.anno.PropertyTemplate;
import com.webaction.anno.PropertyTemplateProperty;
import com.webaction.runtime.components.openprocessor.StriimOpenProcessor;
import com.webaction.runtime.containers.WAEvent;
import com.webaction.runtime.containers.IBatch;
import wa.lstmae.LSTMAEResult_1_0;
import wa.lstmae.LSTMAEPayload_1_0;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Iterator;
import java.util.Map;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

@PropertyTemplate(
    name = "LSTMAEScoreCaller",
    type = AdapterType.process,
    properties = {
        @PropertyTemplateProperty(name = "apiEndpoint", type = String.class, required = false, defaultValue = "http://localhost:8000/v1/score"),
        @PropertyTemplateProperty(name = "timeoutMs", type = Integer.class, required = false, defaultValue = "5000"),
        @PropertyTemplateProperty(name = "maxRetries", type = Integer.class, required = false, defaultValue = "3")
    },
    outputType = LSTMAEPayload_1_0.class,
    inputType  = LSTMAEResult_1_0.class
)
public class LSTMAEScoreCaller extends StriimOpenProcessor {

    private static final Logger logger = LogManager.getLogger(LSTMAEScoreCaller.class);

    private HttpClient httpClient;
    private String apiEndpoint;
    private int timeoutMs;
    private int maxRetries;

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
            logger.info("LSTMAEScoreCaller initialized. Endpoint: {}, Timeout: {}ms, Retries: {}",
                apiEndpoint, effectiveTimeout, effectiveRetries);
        }

        IBatch<WAEvent> batch = getAdded();
        if (batch == null) return;

        Iterator<WAEvent> it = batch.iterator();
        while (it.hasNext()) {
            WAEvent inputEvent = it.next();

            try {
                Object eventData = inputEvent.data;
                String valuesList;
                String windowStart;
                String windowEnd;

                if (eventData != null && eventData.getClass().getName().contains("WAEvent")) {
                    try {
                        java.lang.reflect.Field dataField = eventData.getClass().getField("data");
                        Object innerData = dataField.get(eventData);
                        eventData = innerData;
                    } catch (NoSuchFieldException | IllegalAccessException e) {
                        logger.error("Failed to unwrap WAEvent: {}", e.getMessage());
                    }
                }

                if (eventData instanceof Object[]) {
                    Object[] arr = (Object[]) eventData;
                    String rawValues = arr.length > 0 ? String.valueOf(arr[0]).trim() : "[]";
                    windowStart = arr.length > 1 ? String.valueOf(arr[1]).trim() : "";
                    windowEnd = arr.length > 2 ? String.valueOf(arr[2]).trim() : "";
                    valuesList = normalizeValuesList(rawValues);
                } else {
                    try {
                        java.lang.reflect.Field f0 = eventData.getClass().getField("values_list");
                        java.lang.reflect.Field f1 = eventData.getClass().getField("window_start");
                        java.lang.reflect.Field f2 = eventData.getClass().getField("window_end");
                        String rawValues = String.valueOf(f0.get(eventData)).trim();
                        windowStart = String.valueOf(f1.get(eventData)).trim();
                        windowEnd = String.valueOf(f2.get(eventData)).trim();
                        valuesList = normalizeValuesList(rawValues);
                    } catch (NoSuchFieldException e) {
                        logger.error("Cannot extract fields from: {}", eventData.getClass().getName());
                        continue;
                    }
                }

                JsonObject body = new JsonObject();
                body.add("values", JsonParser.parseString(valuesList));
                body.addProperty("window_start", windowStart);
                body.addProperty("window_end", windowEnd);

                String responseBody = callApiWithRetry(body.toString());

                if (responseBody == null) {
                    logger.error("All retries exhausted for window=[{}, {}]", windowStart, windowEnd);
                    continue;
                }

                if (responseBody.isEmpty()) {
                    // 204 No Content -- non-Sunday window, skip silently
                    continue;
                }

                JsonObject resp = JsonParser.parseString(responseBody).getAsJsonObject();

                String isAnomaly = resp.get("is_anomaly").getAsBoolean() ? "true" : "false";
                String anomalyScore = String.valueOf(resp.get("anomaly_score").getAsDouble());
                String threshold = String.valueOf(resp.get("threshold").getAsDouble());

                logger.info("Scored window=[{}, {}], is_anomaly={}, score={}", windowStart, windowEnd, isAnomaly, anomalyScore);

                LSTMAEResult_1_0 result = new LSTMAEResult_1_0();
                result.is_anomaly = isAnomaly;
                result.anomaly_score = anomalyScore;
                result.threshold = threshold;
                result.window_start = windowStart;
                result.window_end = windowEnd;

                send(result);

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

                if (status == 204) {
                    return "";  // Skip signal -- no content
                }

                if (status >= 200 && status < 300) {
                    return response.body();
                }

                if (status >= 400 && status < 500) {
                    logger.error("Client error: status={}, body={}", status, response.body());
                    return null;
                }

                logger.warn("Server error: status={}, attempt={}/{}", status, attempt, maxRetries);

            } catch (Exception e) {
                logger.warn("Exception calling API: {}, attempt={}/{}", e.getMessage(), attempt, maxRetries);
            }

            try {
                Thread.sleep(backoffMs);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return null;
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
        logger.info("LSTMAEScoreCaller shutting down.");
    }

    @Override
    public Map getAggVec() { return null; }

    @Override
    public void setAggVec(Map aggVec) { }
}
