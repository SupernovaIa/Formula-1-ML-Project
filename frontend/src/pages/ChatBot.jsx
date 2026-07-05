import { useEffect, useRef, useState } from "react";
import { streamChat } from "../api/client";

const MODEL = "gpt-5.4-mini";

const WELCOME = {
  role: "assistant",
  content: "Hi! Ask me anything about the races covered in my reference documents.",
};

export default function ChatBot() {
  const [messages, setMessages] = useState([WELCOME]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    const history = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const stream = await streamChat(history, MODEL);
      const reader = stream.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunkText = decoder.decode(value, { stream: true });
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          updated[updated.length - 1] = { ...last, content: last.content + chunkText };
          return updated;
        });
      }
    } catch (err) {
      setError(err);
      setMessages((prev) => prev.slice(0, -1)); // drop the empty assistant bubble
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <h1>🏎️ F1 Chatbot</h1>

      <div className="chat-window">
        {messages.map((m, i) => (
          <div key={i} className={`chat-bubble ${m.role}`}>
            {m.content || (loading && i === messages.length - 1 && <span className="chat-typing" />)}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {error && <p className="status-text error">{error.message}</p>}

      <form className="chat-input-row" onSubmit={sendMessage}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the races in the knowledge base…"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>Send</button>
      </form>
    </div>
  );
}
