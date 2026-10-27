(() => {
  if (self.crossOriginIsolated || !("serviceWorker" in navigator) || !window.isSecureContext) return;

  const reloadKey = `vegaml-coi-reloaded:${location.pathname}`;
  if (sessionStorage.getItem(reloadKey) === "1") return;

  const reloadForIsolation = () => {
    if (sessionStorage.getItem(reloadKey) === "1") return;
    sessionStorage.setItem(reloadKey, "1");
    location.reload();
  };

  navigator.serviceWorker.register(new URL("./coi-sw.js", document.currentScript.src))
    .then((registration) => {
      if (registration.active && !navigator.serviceWorker.controller) {
        reloadForIsolation();
        return;
      }

      const worker = registration.installing || registration.waiting;
      if (worker) {
        worker.addEventListener("statechange", () => {
          if (worker.state === "activated") reloadForIsolation();
        });
      }
    })
    .catch((error) => console.warn("Cross-origin isolation unavailable; WASM will use one thread:", error));
})();
