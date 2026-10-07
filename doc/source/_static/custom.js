// Cmd+K (Ctrl+K elsewhere) focuses the search box, next to Sphinx's "/".
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    const input = document.querySelector(".searchbox input");
    if (input) {
      event.preventDefault();
      input.focus();
      input.select();
    }
  }
});

// Show the shortcut of the platform in the search box.
document.addEventListener("DOMContentLoaded", () => {
  const kbd = document.querySelector(".searchbox kbd");
  if (kbd) {
    kbd.textContent = /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘K" : "Ctrl K";
  }
});
