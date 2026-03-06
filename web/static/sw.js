// Service Worker for Dinger PWA
const SW_VERSION = 'v4';
const PRECACHE_NAME = `dinger-precache-${SW_VERSION}`;
const RUNTIME_STATIC_CACHE = `dinger-static-${SW_VERSION}`;
const RUNTIME_CDN_CACHE = `dinger-cdn-${SW_VERSION}`;
const OFFLINE_URL = '/static/offline.html';
const CDN_HOSTS = new Set(['cdn.jsdelivr.net', 'cdnjs.cloudflare.com']);

// Install — cache offline page and essential assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(PRECACHE_NAME).then(cache => {
      return cache.addAll([
        OFFLINE_URL,
        '/static/icons/icon-192.png',
        '/static/icons/icon-512.png'
      ]);
    })
  );
  self.skipWaiting();
});

// Activate — clean old caches
self.addEventListener('activate', event => {
  const keep = new Set([PRECACHE_NAME, RUNTIME_STATIC_CACHE, RUNTIME_CDN_CACHE]);
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys
          .filter(k => (k.startsWith('dinger-') || k.startsWith('baseball-')) && !keep.has(k))
          .map(k => caches.delete(k))
      );
    }).then(() => clients.claim())
  );
});

// Fetch — network first, offline fallback for navigation requests
self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;

  const url = new URL(event.request.url);

  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(async () => {
        return caches.match(OFFLINE_URL);
      })
    );
    return;
  }

  if (url.origin === self.location.origin && url.pathname.startsWith('/static/')) {
    event.respondWith(staleWhileRevalidate(event.request, RUNTIME_STATIC_CACHE));
    return;
  }

  if (CDN_HOSTS.has(url.hostname)) {
    event.respondWith(staleWhileRevalidate(event.request, RUNTIME_CDN_CACHE));
  }
});

async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  const networkPromise = fetch(request)
    .then(response => {
      if (response && (response.ok || response.type === 'opaque')) {
        cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => null);

  if (cached) {
    networkPromise.catch(() => null);
    return cached;
  }

  const networkResponse = await networkPromise;
  return networkResponse || new Response('', { status: 504, statusText: 'Gateway Timeout' });
}

// Push notification handler
self.addEventListener('push', event => {
  let data = { title: 'Baseball Alert', body: 'New update available' };

  if (event.data) {
    try {
      data = event.data.json();
    } catch (e) {
      data.body = event.data.text();
    }
  }

  const options = {
    body: data.body || '',
    icon: data.icon || '/static/icons/icon-192.png',
    badge: '/static/icons/icon-192.png',
    tag: data.tag || 'baseball-alert',
    renotify: true,
    data: {
      url: data.url || '/',
      gameId: data.game_id || null,
      removeGame: data.removeGame || false,
    },
    vibrate: [100, 200, 100]
  };

  event.waitUntil(
    self.registration.showNotification(data.title || 'Baseball Alert', options)
      .then(() => {
        // If game is final, tell open clients to remove it from favorites
        if (data.removeGame && data.game_id) {
          return clients.matchAll({ type: 'window', includeUncontrolled: true })
            .then(clientList => {
              clientList.forEach(client => {
                client.postMessage({
                  type: 'dinger:remove-game',
                  gameId: data.game_id,
                });
              });
            });
        }
      })
  );
});

// Notification click — open the relevant page
self.addEventListener('notificationclick', event => {
  event.notification.close();

  const url = event.notification.data.url || '/';

  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true })
      .then(clientList => {
        for (const client of clientList) {
          if (client.url.includes(self.location.origin) && 'focus' in client) {
            client.navigate(url);
            return client.focus();
          }
        }
        return clients.openWindow(url);
      })
  );
});
