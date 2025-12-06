const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");

// Simple random session id
const sessionId = "sess-" + Math.random().toString(36).slice(2);

document.getElementById("show-patients").addEventListener("click", () => {
  fetch(`${BASE_URL}/patients`)
    .then((res) => res.json())
    .then((data) => {
      const box = document.getElementById("patients-list");
      box.style.display = "block";
      box.innerHTML =
        "<strong>Available Patients:</strong><br>" +
        data.patients.map((p) => "- " + p).join("<br>");
    });
});

function appendMessage(text, sender, agent = "system", metadata = {}) {
  const div = document.createElement("div");
  div.classList.add("message");

  if (sender === "user") {
    div.classList.add("user");
  } else {
    div.classList.add(`agent-${agent}`);
  }

  const content = document.createElement("div");
  content.textContent = text;
  div.appendChild(content);

  if (sender !== "user" && metadata) {
    const meta = document.createElement("div");
    meta.classList.add("meta");
    meta.textContent =
      `agent=${agent}` +
      (metadata.used_web_search ? " | used web search" : "") +
      (metadata.num_rag_docs ? ` | RAG docs=${metadata.num_rag_docs}` : "");
    div.appendChild(meta);
  }

  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Initial message from system
appendMessage(
  "Hello! I'm your post-discharge care assistant. Let's get started.",
  "system",
  "system"
);

// First step: ask name via backend stage machine
fetch(`${BASE_URL}/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    session_id: sessionId,
    message: "INIT",
  }),
})
  .then((res) => res.json())
  .then((data) => {
    appendMessage(data.reply, "agent", data.agent, data.metadata || {});
  })
  .catch((err) => console.error(err));

// Handle user input
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;

  appendMessage(text, "user");

  fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message: text,
    }),
  })
    .then((res) => res.json())
    .then((data) => {
      appendMessage(data.reply, "agent", data.agent, data.metadata || {});
    })
    .catch((err) => {
      console.error(err);
      appendMessage(
        "There was an error connecting to the server.",
        "agent",
        "system"
      );
    });

  userInput.value = "";
});
