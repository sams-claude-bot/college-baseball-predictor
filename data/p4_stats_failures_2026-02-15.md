# P4 Stats Collection Failures - February 15, 2026

## Summary
- **Collection Time:** Sunday Feb 15, 2026 @ 11 PM CST (cron job b826db7c)
- **Total Teams:** 68 P4 conference teams
- **Success Rate:** 37/68 (54%)
- **Failed Teams:** 25 (37%)
- **Skipped Teams:** 6 (no baseball programs: SMU, Syracuse, Colorado, Iowa State + Oregon State missing URL)

## Failed Teams by Category

### 404 Errors - Stats Not Published Yet (10 teams)
These sites returned 404 errors - their 2026 stats pages likely aren't live yet.

| Team | Conference | URL Attempted | Notes |
|------|------------|---------------|-------|
| Clemson | ACC | clemsontigers.com | Season just started 2/13, stats not yet published |
| Georgia Tech | ACC | ramblinwreck.com | Season just started 2/13, stats not yet published |
| Stanford | ACC | gostanford.com | Season just started 2/13, stats not yet published |
| Virginia | ACC | virginiasports.com | Season just started 2/13, stats not yet published |
| Arizona | Big 12 | arizonawildcats.com | Season just started 2/13, stats not yet published |
| Arizona State | Big 12 | thesundevils.com | Season just started 2/13, stats not yet published |
| Arkansas | SEC | arkansasrazorbacks.com | Season just started 2/13, stats not yet published |
| Auburn | SEC | auburntigers.com | Season just started 2/13, stats not yet published |
| Kentucky | SEC | ukathletics.com | Season just started 2/13, stats not yet published |
| South Carolina | SEC | gamecocksonline.com | Season just started 2/13, stats not yet published |

### Parsing Issues (9 teams)
These sites returned HTML but the scraper couldn't parse the table format correctly.

| Team | Conference | URL Attempted | Notes |
|------|------------|---------------|-------|
| California | ACC | calbears.com | Non-standard SIDEARM table format |
| Notre Dame | ACC | und.com | Non-standard SIDEARM table format |
| Miami (FL) | ACC | miamihurricanes.com | Uses WordPress cumestats, requires JavaScript |
| Illinois | Big Ten | fightingillini.com | Non-standard SIDEARM table format |
| Iowa | Big Ten | hawkeyesports.com | Non-standard SIDEARM table format |
| Maryland | Big Ten | umterps.com | Non-standard SIDEARM table format |
| Nebraska | Big Ten | huskers.com | Non-standard SIDEARM table format |
| Vanderbilt | SEC | vucommodores.com | Non-standard SIDEARM table format |
| LSU | SEC | lsusports.net | Non-standard SIDEARM table format |

### Other Failures (6 teams)
Various issues including redirects and site configuration problems.

| Team | Conference | URL Attempted | Notes |
|------|------------|---------------|-------|
| Kansas State | Big 12 | kstatesports.com | URL redirects to main page |
| Penn State | Big Ten | gopsusports.com | Stats page structure issue |
| Purdue | Big Ten | purduesports.com | Stats page structure issue |
| Cincinnati | Big 12 | gobearcats.com | Stats page structure issue |
| UCF | Big 12 | ucfknights.com | Stats page structure issue |
| BYU | Big 12 | byucougars.com | Stats page structure issue |

## Successful Teams (37)

### SEC (6/16 = 38%)
✅ Alabama, Florida, Georgia, Missouri, Oklahoma, Ole Miss

### Big Ten (9/18 = 50%)
✅ Indiana, Michigan, Michigan State, Minnesota, Northwestern, Ohio State, Oregon, Rutgers, UCLA, USC, Washington, Wisconsin

### ACC (8/17 = 47%)
✅ Boston College, Duke, Florida State, Louisville, NC State, North Carolina, Pittsburgh, Wake Forest

### Big 12 (9/16 = 56%)
✅ Baylor, Houston, Kansas, Oklahoma State, TCU, Texas Tech, Utah, West Virginia

## Recommendations

1. **404 Errors:** These teams should auto-resolve as their athletics websites publish 2026 stats pages. Retry in Thursday collection.

2. **Parsing Issues:** May need browser-based scraping (Playwright) for these sites. Consider building team-specific parsers for major programs like LSU, Vanderbilt, Notre Dame.

3. **Redirect Issues:** Kansas State needs manual URL investigation - likely different path structure.

4. **Miami Special Case:** Uses WordPress cumestats format, not SIDEARM. Needs dedicated parser with JS rendering.

## Next Collection
- **Scheduled:** Thursday Feb 20, 8:00 AM CST
- **Expected:** Higher success rate as more teams publish 2026 stats
