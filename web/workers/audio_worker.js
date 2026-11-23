'use strict';

// Audio worker placeholder for WebAudio / Opus / XMA2 stream decoding.
self.onmessage = (ev) => {
  if (ev.data && ev.data.init) {
    self.postMessage({ ready: true });
  }
};
