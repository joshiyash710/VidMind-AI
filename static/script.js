document.addEventListener("DOMContentLoaded", () => {
    let sessionId = null;
    let crossVideoMode = false;
    let loadedVideos = [];
    let isReady = false;

    // Elements
    const heroSection = document.getElementById("hero-section");
    const chatSection = document.getElementById("chat-wrapper");
    const urlInput = document.getElementById("url-input");
    const loadBtn = document.getElementById("load-btn");
    const statusMessage = document.getElementById("status-message");
    const closeChatBtn = document.getElementById("close-chat-btn");

    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const chatMessages = document.getElementById("chat-messages");
    const chatHeader = document.querySelector(".chat-header");

    const controlsContainer = document.createElement("div");
    controlsContainer.className = "video-controls";

    function renderVideoControls() {
        if (!chatHeader) return;

        controlsContainer.innerHTML = "";

        const addVideoBtn = document.createElement("button");
        addVideoBtn.className = "tool-btn";
        addVideoBtn.textContent = "Add Video";
        addVideoBtn.title = "Load an additional video in this session";
        addVideoBtn.addEventListener("click", addVideo);

        const crossVideoBtn = document.createElement("button");
        crossVideoBtn.className = "tool-btn";
        crossVideoBtn.id = "crossVideoBtn";
        crossVideoBtn.textContent = `Cross-video: ${crossVideoMode ? "On" : "Off"}`;
        crossVideoBtn.title = "Toggle cross-video QA mode";
        crossVideoBtn.addEventListener("click", toggleCrossVideoMode);

        const exportChatBtn = document.createElement("button");
        exportChatBtn.className = "tool-btn";
        exportChatBtn.textContent = "Export Chat";
        exportChatBtn.title = "Download current conversation";
        exportChatBtn.addEventListener("click", exportChat);

        controlsContainer.appendChild(addVideoBtn);
        controlsContainer.appendChild(crossVideoBtn);
        controlsContainer.appendChild(exportChatBtn);

        const existing = document.getElementById("videoControlsArea");
        if (existing) existing.replaceWith(controlsContainer);
        else {
            controlsContainer.id = "videoControlsArea";
            chatHeader.appendChild(controlsContainer);
        }

        updateCrossVideoButton();
    }

    function updateCrossVideoButton() {
        const btn = document.getElementById("crossVideoBtn");
        if (!btn) return;
        btn.textContent = `Cross-video: ${crossVideoMode ? "On" : "Off"}`;
        btn.style.background = crossVideoMode ? "rgba(31, 162, 145, 0.18)" : "transparent";
        btn.style.color = crossVideoMode ? "#ffffff" : "#dedede";
    }

    function setLoadedVideos(videos) {
        loadedVideos = videos;

        let list = document.getElementById("loadedVideosList");
        if (!list) {
            list = document.createElement("div");
            list.id = "loadedVideosList";
            list.className = "loaded-videos-list";
            if (chatHeader) chatHeader.parentNode.insertBefore(list, chatHeader.nextSibling);
        }

        list.innerHTML = "";
        if (loadedVideos.length === 0) {
            list.innerHTML = "<div class='note'>No videos loaded yet.</div>";
            return;
        }

        loadedVideos.forEach((video, index) => {
            const item = document.createElement("div");
            item.className = "loaded-video-item";
            item.textContent = `${video.label || `Video ${index + 1}`}: ${video.title || video.video_id}`;
            list.appendChild(item);
        });
    }

    function showStatus(message, isError = false) {
        if (!statusMessage) return;
        statusMessage.textContent = message;
        statusMessage.style.color = isError ? "#ffb4ab" : "#a0aab2";
        statusMessage.classList.remove("hidden");
    }

    function appendMessage(sender, text, timestamps = []) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}-msg`;

        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = "msg-bubble";

        let formattedText = text || "";
        if (sender === "ai") {
            formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            formattedText = formattedText.replace(/\n/g, '<br>');

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

    function showTyping() {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message ai-msg typing-indicator-container";
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
        if (indicator) indicator.remove();
    }

    async function addVideo() {
        if (!sessionId) {
            showStatus("Please load a video first", true);
            return;
        }

        const url = prompt("Enter the YouTube URL to add:");
        if (!url || !url.trim()) return;

        showStatus("Adding video...", false);

        try {
            const response = await fetch("/add-video", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId, url: url.trim() })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Failed to add video.");

            loadedVideos.push({ video_id: data.video_id, label: data.label, title: data.title });
            setLoadedVideos(loadedVideos);

            appendMessage("ai", `Added ${data.label}. Cross-video queries now include this video.`);
            showStatus("Video added successfully.");
        } catch (err) {
            showStatus(err.message || "Add video failed", true);
        }
    }

    function toggleCrossVideoMode() {
        crossVideoMode = !crossVideoMode;
        updateCrossVideoButton();
        appendMessage("ai", crossVideoMode ? "Cross-video mode enabled." : "Cross-video mode disabled.");
    }

    async function exportChat() {
        if (!sessionId) {
            showStatus("Please load a video first", true);
            return;
        }

        try {
            const response = await fetch("/export-chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Failed to export chat.");

            const blob = new Blob([data.content || ""], { type: "text/plain" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "vidmind_chat_export.txt";
            a.click();
            URL.revokeObjectURL(url);

            showStatus("Chat exported successfully.");
        } catch (err) {
            showStatus(err.message || "Chat export failed", true);
        }
    }

    function setCrossVideoMode(enabled) {
        crossVideoMode = enabled;
        updateCrossVideoButton();
    }

    async function loadVideo() {
        const url = urlInput.value.trim();
        if (!url) {
            showStatus("Please enter a YouTube URL", true);
            return;
        }

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
            if (!response.ok) throw new Error(data.detail || "Failed to load transcript.");

            sessionId = data.session_id;
            isReady = true;

            loadedVideos = [{ video_id: data.video_id, label: "Video 1", title: data.title || "Video 1" }];
            setLoadedVideos(loadedVideos);
            setCrossVideoMode(false);

            heroSection.classList.add("hidden");
            setTimeout(() => {
                heroSection.style.display = "none";
                if (chatSection) chatSection.classList.remove("hidden");
                if (chatInput) chatInput.focus();
            }, 500);

            if (chatMessages) chatMessages.innerHTML = "";
            appendMessage("ai", "Hello! I am **VidMind AI**. I've successfully analyzed your video. What would you like to know?");
        } catch (error) {
            showStatus(error.message, true);
        } finally {
            urlInput.disabled = false;
            loadBtn.disabled = false;
            loadBtn.innerHTML = "<i class='bx bx-up-arrow-alt'></i>";
        }
    }

    loadBtn.addEventListener("click", loadVideo);
    closeChatBtn.addEventListener("click", () => {
        if (chatSection) {
            chatSection.classList.add("hidden");
            setTimeout(() => {
                heroSection.style.display = "flex";
                setTimeout(() => heroSection.classList.remove("hidden"), 50);
            }, 500);
        }
        urlInput.value = "";
        isReady = false;
    });

    urlInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") loadVideo();
    });

    async function handleChat() {
        const question = chatInput.value.trim();
        if (!question || !isReady) return;

        appendMessage("user", question);
        chatInput.value = "";
        chatInput.disabled = true;
        sendBtn.disabled = true;

        showTyping();

        try {
            const endpoint = crossVideoMode ? "/cross-video-chat" : "/chat";
            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: question, session_id: sessionId })
            });

            const data = await response.json();
            removeTyping();

            if (!response.ok) throw new Error(data.detail || "Failed to get response.");

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
        if (e.key === "Enter") handleChat();
    });

    renderVideoControls();
    setLoadedVideos([]);
});