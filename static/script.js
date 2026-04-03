document.addEventListener("DOMContentLoaded", () => {
    // Session ID from server
    let sessionId = null;
    
    // Elements
    const heroSection = document.getElementById("hero-section");
    const chatSection = document.getElementById("chat-wrapper");
    
    const urlInput = document.getElementById("url-input");
    const loadBtn = document.getElementById("load-btn");
    const statusMessage = document.getElementById("status-message");
    const closeChatBtn = document.getElementById("close-chat-btn");
    
    // Chat interface
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const chatMessages = document.getElementById("chat-messages");

    let isReady = false;

    // Helper to show status msg
    function showStatus(message, isError = false) {
        statusMessage.textContent = message;
        statusMessage.style.color = isError ? "#ffb4ab" : "#a0aab2";
        statusMessage.classList.remove("hidden");
    }

    // Helper to append messages to the chat area
    function appendMessage(sender, text, timestamps = []) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}-msg`;
        
        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = "msg-bubble";
        
        // simple Markdown formatting for AI response (e.g. bolding)
        let formattedText = text;
        if (sender === "ai") {
            formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            formattedText = formattedText.replace(/\n/g, '<br>');
            
            // Build timestamps HTML
            if (timestamps && timestamps.length > 0) {
                let tsHtml = '<div style="margin-top:10px; display:flex; gap:6px; flex-wrap:wrap;">';
                timestamps.forEach(ts => {
                    tsHtml += `<a href="https://www.youtube.com/watch?v=${ts.video_id}&t=${ts.time}s" target="_blank" class="tool-btn" style="text-decoration:none; display:inline-flex; align-items:center; gap:4px; font-family:var(--font-main);">▶ ${ts.label}</a>`;
                });
                tsHtml += '</div>';
                formattedText += tsHtml;
            }
        }

        bubbleDiv.innerHTML = formattedText;
        msgDiv.appendChild(bubbleDiv);
        
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Typing indicator
    function showTyping() {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ai-msg typing-indicator-container`;
        msgDiv.id = "typing-indicator";
        
        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = "msg-bubble";
        bubbleDiv.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        msgDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function removeTyping() {
        const indicator = document.getElementById("typing-indicator");
        if (indicator) {
            indicator.remove();
        }
    }

    // Load Transcript logic
    loadBtn.addEventListener("click", async () => {
        const url = urlInput.value.trim();
        if (!url) {
            showStatus("Please enter a YouTube URL", true);
            return;
        }

        // Processing state
        urlInput.disabled = true;
        loadBtn.disabled = true;
        loadBtn.innerHTML = "<i class='bx bx-loader-alt bx-spin'></i>";
        statusMessage.classList.add("hidden");

        try {
            const response = await fetch("/load-video", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url: url })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Failed to load transcript.");
            }

            // SUCCESS! Transition UI to Chat
            sessionId = data.session_id;
            isReady = true;
            
            // Hide Hero, Show Chat
            heroSection.classList.add("hidden");
            // small delay to let hero fade out before chat fades in
            setTimeout(() => {
                heroSection.style.display = 'none';
                chatSection.classList.remove("hidden");
                chatInput.focus();
            }, 500);
            
            chatMessages.innerHTML = '';
            appendMessage("ai", "Hello! I am **VidMind AI**. I've successfully analyzed your video. What would you like to know?");
            
        } catch (error) {
            showStatus(error.message, true);
        } finally {
            urlInput.disabled = false;
            loadBtn.disabled = false;
            loadBtn.innerHTML = "<i class='bx bx-up-arrow-alt'></i>";
        }
    });

    // Close / Reset Chat
    closeChatBtn.addEventListener("click", () => {
        chatSection.classList.add("hidden");
        setTimeout(() => {
            heroSection.style.display = 'flex';
            // slight delay to allow display computation
            setTimeout(() => {
                heroSection.classList.remove("hidden");
            }, 50);
        }, 500);
        urlInput.value = "";
        isReady = false;
    });

    // Handle pressing enter on URL input
    urlInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            loadBtn.click();
        }
    });

    // Chat Logic
    async function handleChat() {
        const question = chatInput.value.trim();
        if (!question || !isReady) return;

        appendMessage("user", question);
        chatInput.value = "";
        
        chatInput.disabled = true;
        sendBtn.disabled = true;

        showTyping();

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: question, session_id: sessionId })
            });

            const data = await response.json();
            removeTyping();

            if (!response.ok) {
                throw new Error(data.detail || "Failed to get response.");
            }

            appendMessage("ai", data.answer, data.timestamps);

        } catch (error) {
            removeTyping();
            appendMessage("ai", `Error: ${error.message}`);
        } finally {
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.focus();
        }
    }

    sendBtn.addEventListener("click", handleChat);
    chatInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            handleChat();
        }
    });
});
