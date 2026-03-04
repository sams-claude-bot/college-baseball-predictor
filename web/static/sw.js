// Service Worker for College Baseball Predictor PWA
const CACHE_NAME = 'baseball-v2';
const OFFLINE_URL = '/static/offline.html';

// Install — cache offline page and essential assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
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
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
      );
    }).then(() => clients.claim())
  );
});

// Fetch — network first, offline fallback for navigation requests
self.addEventListener('fetch', event => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(() => {
        return caches.match(OFFLINE_URL);
      })
    );
  }
});

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
      url: data.url || '/'
    },
    vibrate: [100, 200, 100]
  };

  event.waitUntil(
    self.registration.showNotification(data.title || 'Baseball Alert', options)
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
