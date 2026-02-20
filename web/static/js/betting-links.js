// Make betting cards and table rows clickable to open game detail
document.addEventListener('DOMContentLoaded', function() {
    // Cards: find all game cards by looking for team-link anchors
    document.querySelectorAll('.card.h-100').forEach(function(card) {
        // Find the game ID from team links inside the card
        var links = card.querySelectorAll('a.team-link');
        if (links.length >= 2) {
            // Get the away and home team IDs from href
            var awayId = links[0].getAttribute('href').replace('/team/', '');
            var homeId = links[1].getAttribute('href').replace('/team/', '');
            // Find the date
            var dateEl = card.querySelector('small.text-muted');
            var date = dateEl ? dateEl.textContent.trim() : '';
            if (date && awayId && homeId) {
                var gameId = date + '_' + awayId + '_' + homeId;
                card.style.cursor = 'pointer';
                card.addEventListener('click', function(e) {
                    if (e.target.tagName === 'A') return;
                    window.location = '/game/' + gameId;
                });
            }
        }
    });

    // Table rows: same approach
    var table = document.getElementById('allGamesTable');
    if (table) {
        table.querySelectorAll('tbody tr').forEach(function(row) {
            var links = row.querySelectorAll('a.team-link');
            if (links.length >= 2) {
                var awayId = links[0].getAttribute('href').replace('/team/', '');
                var homeId = links[1].getAttribute('href').replace('/team/', '');
                var dateEl = row.closest('table') ? null : null;
                // For table rows, we need the date from somewhere
                // Use the card section date or just link to a search
                row.style.cursor = 'pointer';
                row.addEventListener('click', function(e) {
                    if (e.target.tagName === 'A') return;
                    // Navigate to game page - construct ID from today's date
                    var today = new Date().toISOString().split('T')[0];
                    var gameId = today + '_' + awayId + '_' + homeId;
                    window.location = '/game/' + gameId;
                });
            }
        });
    }
});
