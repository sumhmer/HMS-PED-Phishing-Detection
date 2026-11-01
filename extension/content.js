// Content script: runs on mail.google.com and extracts the open email.
// Original version: base paper code, CMPE209FinalProject (E. Pham, T. Bui, H. Chen, IEEE SVCC 2025).
// Change in this project: the popup now ASKS for the email when it opens
// (GET_EMAIL message) instead of waiting for a DOM change, which fixes the
// "Loading email..." hang in the original extension.

console.log("Hybrid phishing scanner: content script running");

function readOpenEmail() {
  const subject = document.querySelector("h2.hP")?.innerText || "";
  const sender = document.querySelector("span.gD")?.getAttribute("email") || "";
  const body = document.querySelector("div.a3s")?.innerText || "";
  if (!body.trim()) return null;
  return { sender, subject, body };
}

// 1) Answer the popup when it asks for the current email.
chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
  if (request.type === "GET_EMAIL") {
    sendResponse({ payload: readOpenEmail() });
  }
});

// 2) Keep the original behaviour too: push the email when Gmail's page changes.
let timeoutId;
const observer = new MutationObserver(() => {
  clearTimeout(timeoutId);
  timeoutId = setTimeout(() => {
    const email = readOpenEmail();
    if (email && email.body.trim().length > 50) {
      // .catch: when the popup is closed there is no receiver, which is fine
      chrome.runtime.sendMessage({ type: "EMAIL_DATA", payload: email }).catch(() => {});
    }
  }, 500);
});
observer.observe(document.body, { childList: true, subtree: true });
