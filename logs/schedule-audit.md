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
