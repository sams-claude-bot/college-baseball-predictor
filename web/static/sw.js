// Service Worker for College Baseball Predictor PWA
const CACHE_NAME = 'baseball-v1';

// Install — cache essential assets
self.addEventListener('install', event => {
  self.skipWaiting();
});

// Activate — clean old caches
self.addEventListener('activate', event => {
  event.waitUntil(clients.claim());
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
    // Vibrate pattern: short-long-short
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
        // Focus existing window if possible
        for (const client of clientList) {
          if (client.url.includes(self.location.origin) && 'focus' in client) {
            client.navigate(url);
            return client.focus();
          }
        }
        // Otherwise open a new window
        return clients.openWindow(url);
      })
  );
});
