const cacheName = 'Bulls&Cows';
const contentToCache = [
  '/html/Bulls&Cows/',
  '/html/Bulls&Cows/index.html',
  '/html/Bulls&Cows/sw.js',
  '/html/Bulls&Cows/manifest.json',
  '/html/Bulls&Cows/favicon.png'
];

// Install the Service Worker
self.addEventListener('install', (e) => {
  e.waitUntil((async () => {
    const cache = await caches.open(cacheName);
    await cache.addAll(contentToCache);
  })());
});

// Fetch content using the Service Worker
self.addEventListener('fetch', (e) => {
  e.respondWith((async () => {
    const r = await caches.match(e.request);
    if (r) return r;
    const response = await fetch(e.request);
    const cache = await caches.open(cacheName);
    cache.put(e.request, response.clone());
    return response;
  })());
});

// Clear the cache
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((keyList) => {
    return Promise.all(keyList.map((key) => {
      if (key === cacheName) { return; }
      return caches.delete(key);
    }))
  }));
});