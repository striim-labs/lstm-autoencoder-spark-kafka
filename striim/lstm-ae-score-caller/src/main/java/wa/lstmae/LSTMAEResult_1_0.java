package wa.lstmae;

import com.webaction.event.SimpleEvent;
import com.webaction.event.WactionConvertible;
import com.webaction.uuid.UUID;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.KryoSerializable;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.Serializable;
import java.util.Map;

/**
 * Hand-built Striim type class for the LSTM-AE result stream.
 *
 * Matches the Striim type:
 *   CREATE TYPE LSTMAEResult (
 *     is_anomaly    String,
 *     anomaly_score String,
 *     threshold     String,
 *     window_start  String,
 *     window_end    String
 *   );
 *
 * Package wa.lstmae matches the lstmae namespace.
 * Bundled inside the OP JAR so no external types JAR is needed.
 */
public class LSTMAEResult_1_0 extends SimpleEvent
        implements WactionConvertible, Serializable, KryoSerializable {

    private static final long serialVersionUID = 1L;
    public static ObjectMapper mapper = newMapper();

    // Fields (match CREATE TYPE field order)
    public String is_anomaly;
    public String anomaly_score;
    public String threshold;
    public String window_start;
    public String window_end;

    // Constructors
    public LSTMAEResult_1_0() { super(); }
    public LSTMAEResult_1_0(long timeStamp) { super(timeStamp); }

    // Getters / Setters
    public String getIs_anomaly() { return is_anomaly; }
    public void setIs_anomaly(String val) { is_anomaly = val; }
    public String getAnomaly_score() { return anomaly_score; }
    public void setAnomaly_score(String val) { anomaly_score = val; }
    public String getThreshold() { return threshold; }
    public void setThreshold(String val) { threshold = val; }
    public String getWindow_start() { return window_start; }
    public void setWindow_start(String val) { window_start = val; }
    public String getWindow_end() { return window_end; }
    public void setWindow_end(String val) { window_end = val; }

    // Payload
    public Object[] getPayload() {
        return new Object[] { is_anomaly, anomaly_score, threshold, window_start, window_end };
    }

    public void setPayload(Object[] payload) {
        if (payload != null && payload.length >= 5) {
            is_anomaly    = (String) payload[0];
            anomaly_score = (String) payload[1];
            threshold     = (String) payload[2];
            window_start  = (String) payload[3];
            window_end    = (String) payload[4];
        }
    }

    // Kryo serialization
    public void write(Kryo kryo, Output output) {
        output.writeString(is_anomaly);
        output.writeString(anomaly_score);
        output.writeString(threshold);
        output.writeString(window_start);
        output.writeString(window_end);
    }

    public void read(Kryo kryo, Input input) {
        is_anomaly    = input.readString();
        anomaly_score = input.readString();
        threshold     = input.readString();
        window_start  = input.readString();
        window_end    = input.readString();
    }

    // WactionConvertible
    @SuppressWarnings("unchecked")
    public boolean setFromContextMap(Map map) {
        Object ts = map.get("timestamp");
        if (ts instanceof Long) {
            this.setTimeStamp(((Long) ts).longValue());
        }
        Object uid = map.get("uuid");
        if (uid instanceof UUID) {
            this._wa_SimpleEvent_ID = (UUID) uid;
        }
        Object k = map.get("key");
        if (k != null) {
            this.key = k.toString();
        }
        Object v;
        v = map.get("context-is_anomaly");
        if (v != null) is_anomaly = v.toString();
        v = map.get("context-anomaly_score");
        if (v != null) anomaly_score = v.toString();
        v = map.get("context-threshold");
        if (v != null) threshold = v.toString();
        v = map.get("context-window_start");
        if (v != null) window_start = v.toString();
        v = map.get("context-window_end");
        if (v != null) window_end = v.toString();
        return true;
    }

    @SuppressWarnings("unchecked")
    public void convertFromWactionToEvent(
            long timeStamp, UUID id, String key, Map map) {
        this.setTimeStamp(timeStamp);
        if (id != null) this.setID(id);
        if (key != null) this.setKey(key);
        if (map != null) setFromContextMap(map);
    }

    public LSTMAEResult_1_0 convertToDeleteEvent() {
        LSTMAEResult_1_0 del = new LSTMAEResult_1_0();
        del.is_anomaly = null;
        del.anomaly_score = null;
        del.threshold = null;
        del.window_start = null;
        del.window_end = null;
        return del;
    }

    // JSON
    public Object fromJSON(String json) {
        try { return mapper.readValue(json, this.getClass()); }
        catch (Exception e) { return null; }
    }

    public String toJSON() {
        try { return mapper.writeValueAsString(this); }
        catch (Exception e) { return null; }
    }

    public String toString() {
        return "LSTMAEResult_1_0{"
            + "is_anomaly=" + is_anomaly
            + ", anomaly_score=" + anomaly_score
            + ", threshold=" + threshold
            + ", window_start=" + window_start
            + ", window_end=" + window_end
            + "}";
    }

    private static ObjectMapper newMapper() {
        try {
            java.lang.reflect.Method m =
                com.webaction.event.ObjectMapperFactory.class
                    .getMethod("getFullInstance");
            return (ObjectMapper) m.invoke(null);
        } catch (Exception e1) {
            try {
                java.lang.reflect.Method m =
                    com.webaction.event.ObjectMapperFactory.class
                        .getMethod("getInstance");
                return (ObjectMapper) m.invoke(null);
            } catch (Exception e2) {
                return new ObjectMapper();
            }
        }
    }
}
