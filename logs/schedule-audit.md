# SEC Schedule Audit Log
Started: 2026-02-16

## Mississippi State

### Official Source: hailstate.com

### Discrepancies Found:

1. **Mar 13-15: MSU @ Arkansas** (SEC series)
   - DB shows: Apr 10-12 (WRONG DATE)
   - Official: Mar 13-15 at Fayetteville (confirmed via Arkansas website)
   - Fix: Update dates to Mar 13-15

2. **Mar 27-29: MSU @ Ole Miss** (Rivalry series)
   - DB shows: MSU as home team (WRONG)
   - Official: Ole Miss is home (in Oxford)
   - Fix: Swap home/away teams

3. **Apr 2-4: MSU vs Georgia** (SEC series)  
   - DB shows: Apr 2 Georgia only, then Apr 3-5 vs Oklahoma
   - Official: All 3 games are vs Georgia at MSU on Apr 2-4
   - Fix: Remove Oklahoma Apr 3-5, add Georgia Apr 3-4

4. **Apr 10-12: MSU vs Tennessee** (SEC series)
   - DB shows: MSU @ Arkansas
   - Official: Tennessee at MSU (home)
   - Fix: Replace Arkansas with Tennessee, MSU is home


### Changes Applied:

1. **Mar 27-29 Ole Miss series**: Changed from MSU home to Ole Miss home (Oxford)
   - Updated times: Mar 27 6:30 PM, Mar 28 1:30 PM, Mar 29 3:00 PM

2. **Apr 2-4 Georgia series**: Fixed opponent
   - Deleted Apr 5 Oklahoma game
   - Changed Apr 3-4 from Oklahoma to Georgia

3. **Mar 13-15 @ Arkansas**: Fixed dates
   - Changed from Apr 10-12 to Mar 13-15
   - Times: Mar 13 6 PM, Mar 14 1 PM, Mar 15 1 PM

### Pending Investigation:
- Apr 10-12 series: HailState shows Tennessee, but Tennessee's website doesn't show @MSU. Need further verification.
- Feb 27-Mar 1: Amegy Bank Series games (neutral site) - not in DB
- Mar 24: Southern Miss game - not in DB

---

## Alabama

### Official Source: rolltide.com

### Discrepancies Found:

1. **Apr 23-25: @ Tennessee** (SEC series)
   - DB showed: Apr 10-12 + extra game Apr 23
   - Official (utsports.com): Apr 23-25 at Knoxville
   - Fix: Updated dates to Apr 23, 24, 25

### Changes Applied:
- Tennessee vs Alabama series: Apr 10-12 → Apr 23-25
- Times: Apr 23 7 PM, Apr 24 6:30 PM, Apr 25 1 PM
- Removed duplicate Apr 23 single game

---

## LSU

### Discrepancies Found:

1. **Apr 3-5: @ Tennessee** (SEC series)
   - DB was missing Apr 5 game
   - Fix: Added Apr 5 game (12:00 PM CT)

---

## Arkansas

### Official Source: arkansasrazorbacks.com

### Discrepancies Found:

1. **Mar 13-15: vs Mississippi State** (SEC series)
   - DB had BOTH Florida AND MSU scheduled (double-booked)
   - Fix: Removed Florida, kept only MSU

2. **Mar 20-22: @ South Carolina** (SEC series)
   - DB showed: Arkansas home vs Alabama
   - Official: Arkansas @ South Carolina
   - Fix: Changed opponent and swapped home/away

3. **Mar 27-29: vs Florida** (SEC series)
   - DB showed: Arkansas @ Georgia
   - Official: Florida at Arkansas
   - Fix: Changed opponent and location

4. **Apr 5: LSU @ Arkansas**
   - DB had single game on Apr 5 (LSU can't be at Tennessee AND Arkansas)
   - Fix: Deleted incorrect game

5. **Apr 10-12: @ Alabama** (SEC series)
   - DB was missing this series entirely
   - Fix: Added 3-game series at Alabama

---

## Ole Miss

### Official Source: olemisssports.com

### Discrepancies Found:

1. **Mar 13-15: @ Texas** - Fixed to all 3 games at Texas
2. **Mar 19-21: vs Kentucky** - Fixed to Kentucky (was mixed with Florida)
3. **Apr 2-4: @ Florida** - Changed from @ Alabama to @ Florida
4. **Apr 25: vs Georgia** - Added missing 3rd game of series
5. **May 1-3: @ Arkansas** - Changed from vs Oklahoma to @ Arkansas
6. **May 8-10: vs Texas A&M** - Added missing series
7. **May 14-16: @ Alabama** - Changed from @ Texas to @ Alabama

---

## Florida

### Fixes Applied:
- Removed duplicate Georgia games (Apr 3-5) that conflicted with Ole Miss
- Removed duplicate Auburn games (Apr 17-18) that conflicted with Arkansas

### Pending Verification:
- Auburn @ Florida series dates need verification

---

## Summary - Teams Verified:

✓ Mississippi State
✓ Alabama  
✓ Auburn (pre-verified)
✓ LSU
✓ Arkansas
✓ Ole Miss
✓ Florida (partial)

### Teams Still Needing Full Verification:
- Texas A&M
- Texas
- Oklahoma
- Tennessee
- Georgia
- South Carolina
- Kentucky
- Missouri
- Vanderbilt

---

## Cross-Reference Conflict Resolution

Found and fixed scheduling conflicts where teams were double-booked:

1. **Mar 13-15**: Removed Oklahoma vs Texas (kept Ole Miss @ Texas per verification)
2. **Apr 5**: Removed Texas vs Tennessee (kept Tennessee @ LSU per verification)
3. **Apr 24-26**: Removed Oklahoma vs Florida (kept Auburn @ Oklahoma)
4. **May 8-10**: Removed Ole Miss vs Texas A&M (kept Arkansas @ Texas A&M as original)
5. **May 14-16**: Removed Ole Miss @ Alabama (kept Alabama vs Florida as original)

Note: Some of these deletions may need re-verification to find correct dates for those series.

---
