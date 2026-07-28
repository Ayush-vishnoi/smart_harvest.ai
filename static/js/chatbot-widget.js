(() => {
    "use strict";

    const history = [];
    let waiting = false;

    function element(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text) node.textContent = text;
        return node;
    }

    function initChatbot() {
        const root = element("aside", "sh-chatbot");
        root.setAttribute("aria-label", "Smart Harvest farming assistant");

        const panel = element("section", "sh-chatbot__panel");
        panel.id = "sh-chatbot-panel";
        panel.setAttribute("role", "dialog");
        panel.setAttribute("aria-modal", "false");
        panel.setAttribute("aria-labelledby", "sh-chatbot-title");
        panel.setAttribute("aria-hidden", "true");

        const header = element("header", "sh-chatbot__header");
        const titleWrap = element("div", "sh-chatbot__title-wrap");
        const icon = element("span", "sh-chatbot__header-icon", "🌱");
        icon.setAttribute("aria-hidden", "true");
        const headingGroup = element("div", "");
        const title = element("h2", "sh-chatbot__title", "Farming Assistant");
        title.id = "sh-chatbot-title";
        headingGroup.append(title, element("p", "sh-chatbot__subtitle", "Smart Harvest AI"));
        titleWrap.append(icon, headingGroup);

        const close = element("button", "sh-chatbot__close", "×");
        close.type = "button";
        close.setAttribute("aria-label", "Close farming assistant");
        header.append(titleWrap, close);

        const messages = element("div", "sh-chatbot__messages");
        messages.setAttribute("role", "log");
        messages.setAttribute("aria-live", "polite");
        messages.setAttribute("aria-relevant", "additions");

        const welcome = element(
            "div",
            "sh-chatbot__bubble sh-chatbot__bubble--assistant",
            "Hello! Ask me about crop care, soil, irrigation, yield, pests, or plant diseases."
        );
        messages.append(welcome);

        const form = element("form", "sh-chatbot__form");
        const input = document.createElement("textarea");
        input.className = "sh-chatbot__input";
        input.rows = 1;
        input.maxLength = 2000;
        input.placeholder = "Ask a farming question…";
        input.setAttribute("aria-label", "Farming question");
        const send = element("button", "sh-chatbot__send", "Send");
        send.type = "submit";
        form.append(input, send);

        const launcher = element("button", "sh-chatbot__launcher", "🌱");
        launcher.type = "button";
        launcher.setAttribute("aria-label", "Open farming assistant");
        launcher.setAttribute("aria-controls", panel.id);
        launcher.setAttribute("aria-expanded", "false");

        panel.append(header, messages, form);
        root.append(panel, launcher);
        document.body.append(root);

        function scrollToLatest() {
            messages.scrollTop = messages.scrollHeight;
        }

        function setOpen(open) {
            root.classList.toggle("sh-chatbot--open", open);
            panel.setAttribute("aria-hidden", String(!open));
            launcher.setAttribute("aria-expanded", String(open));
            launcher.setAttribute("aria-label", open ? "Close farming assistant" : "Open farming assistant");
            if (open) window.setTimeout(() => input.focus(), 180);
            else launcher.focus();
        }

        function addMessage(role, content, sources = []) {
            const item = element("div", `sh-chatbot__message sh-chatbot__message--${role}`);
            const bubble = element("div", `sh-chatbot__bubble sh-chatbot__bubble--${role}`, content);
            item.append(bubble);
            if (role === "assistant" && sources.length) {
                item.append(element("div", "sh-chatbot__sources", `Sources: ${sources.join(", ")}`));
            }
            messages.append(item);
            scrollToLatest();
            return item;
        }

        launcher.addEventListener("click", () => setOpen(!root.classList.contains("sh-chatbot--open")));
        close.addEventListener("click", () => setOpen(false));
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape" && root.classList.contains("sh-chatbot--open")) setOpen(false);
        });
        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                form.requestSubmit();
            }
        });

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            const message = input.value.trim();
            if (!message || waiting) return;

            const priorHistory = history.slice(-12);
            addMessage("user", message);
            input.value = "";
            waiting = true;
            send.disabled = true;
            input.disabled = true;

            const typing = element("div", "sh-chatbot__typing");
            typing.setAttribute("aria-label", "Assistant is typing");
            typing.append(element("span"), element("span"), element("span"));
            messages.append(typing);
            scrollToLatest();

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({message, history: priorHistory}),
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok) throw new Error(data.error || "Unable to contact the farming assistant.");
                typing.remove();
                addMessage("assistant", data.reply, Array.isArray(data.sources) ? data.sources : []);
                history.push({role: "user", content: message}, {role: "assistant", content: data.reply});
                if (history.length > 24) history.splice(0, history.length - 24);
            } catch (error) {
                typing.remove();
                addMessage("assistant", error.message || "The farming assistant is temporarily unavailable. Please try again.");
            } finally {
                waiting = false;
                send.disabled = false;
                input.disabled = false;
                input.focus();
            }
        });
    }

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initChatbot);
    else initChatbot();
})();
