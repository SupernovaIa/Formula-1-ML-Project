import { useEffect, useRef, useState } from "react";
import Icon from "../components/Icon";
import PageHead from "../components/PageHead";
import { streamChat } from "../api/client";

const MODEL = "gpt-5.4-mini";

const SUGGESTIONS = ["Who won the race?", "How did the race unfold?", "What were the key moments?"];

const WELCOME = {
  role: "assistant",
  content: "Hi! Ask me about the 2024 Australian Grand Prix — who won, how it played out, key moments.",
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

  function sendMessage(e) {
    e.preventDefault();
    ask(input);
  }

  async function ask(question) {
    const text = question.trim();
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
      <PageHead
        label="2024 Australian Grand Prix"
        title="Ask the"
        em="race"
        lede="Ask about the 2024 Australian Grand Prix in your own words."
      />

      <div className="a-panel chat-window" aria-live="polite">
        {messages.map((m, i) => (
          <div key={i} className={`chat-bubble ${m.role}`}>
            {m.content || (loading && i === messages.length - 1 && <span className="chat-typing" />)}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {messages.length === 1 && (
        <div className="chat-suggestions">
          {SUGGESTIONS.map((q) => (
            <button key={q} type="button" className="a-chip chat-chip" onClick={() => ask(q)}>
              {q}
            </button>
          ))}
        </div>
      )}

      {error && (
        <p className="status status--error" role="alert">
          <Icon name="circle-alert" />
          {error.message}
        </p>
      )}

      <form className="chat-input-row" onSubmit={sendMessage}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g. Who won the race?"
          aria-label="Your question"
          disabled={loading}
        />
        <button type="submit" className="a-btn a-btn--primary" disabled={loading || !input.trim()}>
          Send <Icon name="arrow-right" />
        </button>
      </form>
    </div>
  );
}
