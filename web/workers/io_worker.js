'use strict';

// HTTP Range (206 Partial Content) VFS worker + bootset prefetch.
let dataBase = '/data/';
let manifest = null;

async function readRange(relPath, offset, length) {
  const end = offset + length - 1;
  const res = await fetch(`${dataBase}${relPath}`, {
    headers: { Range: `bytes=${offset}-${end}` },
  });
  if (!res.ok && res.status !== 206) {
    throw new Error(`HTTP ${res.status} reading ${relPath}[${offset}..${end}]`);
  }
  return new Uint8Array(await res.arrayBuffer());
}

async function init(base) {
  dataBase = base.endsWith('/') ? base : `${base}/`;
  let summary = 'no data-manifest.json yet';
  try {
    const res = await fetch(`${dataBase}data-manifest.json`);
    if (res.ok) {
      manifest = await res.json();
      summary = `${manifest.file_count || 0} files indexed`;
    }
  } catch {
    // Manifest is optional until the user generates local-data/data-manifest.json.
  }
  self.postMessage({ ready: true, summary });
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};
  try {
    if (msg.init) {
      await init(msg.dataBase || '/data/');
    } else if (msg.read) {
      const bytes = await readRange(msg.path, msg.offset, msg.length);
      self.postMessage({ id: msg.id, bytes }, [bytes.buffer]);
    }
  } catch (err) {
    self.postMessage({ id: msg.id, error: String(err) });
  }
};
