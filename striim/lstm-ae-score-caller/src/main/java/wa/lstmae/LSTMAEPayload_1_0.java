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
 * Hand-built Striim type class for the LSTM-AE payload stream.
 *
 * Matches the Striim type:
 *   CREATE TYPE LSTMAEPayload (
 *     values_list  String,
 *     window_start java.lang.String,
 *     window_end   java.lang.String
 *   );
 *
 * Package wa.lstmae matches the lstmae namespace.
 */
public class LSTMAEPayload_1_0 extends SimpleEvent
        implements WactionConvertible, Serializable, KryoSerializable {

    private static final long serialVersionUID = 1L;
    public static ObjectMapper mapper = newMapper();

    // Fields (match CREATE TYPE field order)
    public String values_list;
    public String window_start;
    public String window_end;

    // Constructors
    public LSTMAEPayload_1_0() { super(); }
    public LSTMAEPayload_1_0(long timeStamp) { super(timeStamp); }

    // Getters / Setters
    public String getValues_list() { return values_list; }
    public void setValues_list(String val) { values_list = val; }
    public String getWindow_start() { return window_start; }
    public void setWindow_start(String val) { window_start = val; }
    public String getWindow_end() { return window_end; }
    public void setWindow_end(String val) { window_end = val; }

    // Payload
    public Object[] getPayload() {
        return new Object[] { values_list, window_start, window_end };
    }

    public void setPayload(Object[] payload) {
        if (payload != null && payload.length >= 3) {
            values_list  = (String) payload[0];
            window_start = (String) payload[1];
            window_end   = (String) payload[2];
        }
    }

    // Kryo serialization
    public void write(Kryo kryo, Output output) {
        output.writeString(values_list);
        output.writeString(window_start);
        output.writeString(window_end);
    }

    public void read(Kryo kryo, Input input) {
        values_list  = input.readString();
        window_start = input.readString();
        window_end   = input.readString();
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
        v = map.get("context-values_list");
        if (v != null) values_list = v.toString();
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

    public LSTMAEPayload_1_0 convertToDeleteEvent() {
        LSTMAEPayload_1_0 del = new LSTMAEPayload_1_0();
        del.values_list = null;
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
        return "LSTMAEPayload_1_0{"
            + "values_list=" + values_list
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
