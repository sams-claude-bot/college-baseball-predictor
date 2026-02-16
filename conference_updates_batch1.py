#!/usr/bin/env python3
"""
Batch updates for conference cleanup - first batch with known teams.
"""

import sqlite3
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'

# Teams I can identify confidently based on athletics knowledge:
BATCH_UPDATES = [
    # Team ID, Conference, Athletics URL
    ('alma', 'Non-D1', 'https://almascots.com'),  # MIAA D3
    ('american-university', 'Patriot', 'https://www.american.edu/athletics/'),  # Patriot League D1
    ('belmont', 'MVC', 'https://www.belmontsports.com'),  # Missouri Valley D1
    ('boise-state', 'MWC', 'https://www.broncosports.com'),  # Mountain West D1
    ('boston-university', 'Patriot', 'https://www.goterriers.com'),  # Patriot League D1
    ('bryant', 'America East', 'https://bryantbulldogs.com'),  # America East D1 (moved from NEC)
    ('buffalo', 'MAC', 'https://www.ubbulls.com'),  # Mid-American D1
    ('cal-baptist', 'WAC', 'https://www.calbaptistlancers.com'),  # Western Athletic D1
    ('colorado-state', 'MWC', 'https://www.csurams.com'),  # Mountain West D1
    ('connecticut', 'Big East', 'https://www.uconnhuskies.com'),  # Big East D1
    ('denver', 'Summit', 'https://www.denverpioneers.com'),  # Summit League D1
    ('detroit-mercy', 'Horizon', 'https://www.detroittitans.com'),  # Horizon League D1
    ('drake', 'MVC', 'https://www.godrakebulldogs.com'),  # Missouri Valley D1
    ('duquesne', 'A-10', 'https://www.duquesnedukes.com'),  # Atlantic 10 D1
    ('eastern-washington', 'Big Sky', 'https://www.goeags.com'),  # Big Sky D1
    ('florida-am', 'SWAC', 'https://www.famuathletics.com'),  # SWAC D1
    ('grambling', 'SWAC', 'https://www.grambling.edu/athletics/'),  # SWAC D1
    ('green-bay', 'Horizon', 'https://www.gbphoenix.com'),  # Horizon League D1
    ('hampton', 'CAA', 'https://www.hamptonpirates.com'),  # Coastal Athletic D1
    ('hartford', 'America East', 'https://www.hartfordhawks.com'),  # America East D1
    ('idaho', 'Big Sky', 'https://www.govandals.com'),  # Big Sky D1
    ('idaho-state', 'Big Sky', 'https://www.isubengals.com'),  # Big Sky D1
    ('iu-indianapolis', 'Horizon', 'https://www.iujags.com'),  # Horizon League D1
    ('little-rock', 'OVC', 'https://www.trojanshare.com'),  # Ohio Valley D1
    ('long-island-university', 'NEC', 'https://www.liusharks.com'),  # Northeast D1
    ('loyola-chicago', 'A-10', 'https://www.loyolaramblers.com'),  # Atlantic 10 D1
    ('loyola-maryland', 'Patriot', 'https://www.loyolagreyhounds.com'),  # Patriot League D1
    ('marquette', 'Big East', 'https://www.gomarquette.com'),  # Big East D1
    ('montana', 'Big Sky', 'https://www.gogriz.com'),  # Big Sky D1
    ('montana-state', 'Big Sky', 'https://www.msubobcats.com'),  # Big Sky D1
    ('new-hampshire', 'America East', 'https://www.unhwildcats.com'),  # America East D1
    ('new-orleans', 'Southland', 'https://www.unoprivateers.com'),  # Southland D1
    ('north-carolina-central', 'MEAC', 'https://www.nccu.edu/athletics/'),  # MEAC D1
    ('north-dakota-fighting', 'Summit', 'https://www.fightinghawks.com'),  # Summit League D1
    ('northern-arizona', 'Big Sky', 'https://www.nauathletics.com'),  # Big Sky D1
    ('northern-iowa', 'MVC', 'https://www.unipanthers.com'),  # Missouri Valley D1
    ('pacific', 'WCC', 'https://www.pacifictigers.com'),  # West Coast D1
    ('pennsylvania', 'Ivy', 'https://www.pennathletics.com'),  # Ivy League D1
    ('portland-state', 'Big Sky', 'https://www.goviks.com'),  # Big Sky D1
    ('robert-morris', 'Horizon', 'https://www.rmcolonials.com'),  # Horizon League D1
    ('sacramento-state', 'Big Sky', 'https://www.hornetsports.com'),  # Big Sky D1
    ('savannah-state', 'MEAC', 'https://www.ssutigers.com'),  # MEAC D1
    ('se-louisiana', 'Southland', 'https://www.lionsports.net'),  # Southland D1
    ('south-carolina-upstate', 'Big South', 'https://www.upstatespartans.com'),  # Big South D1
    ('south-dakota', 'Summit', 'https://www.goyotes.com'),  # Summit League D1
    ('st-francis-bkn', 'NEC', 'https://www.sfuterriers.com'),  # Northeast D1
    ('texas-aandm-corpus-christi', 'Southland', 'https://www.goislanders.com'),  # Southland D1
    ('uic', 'Horizon', 'https://www.uicflames.com'),  # Horizon League D1
    ('ul-lafayette', 'Sun Belt', 'https://www.ragincajuns.com'),  # Sun Belt D1
    ('umass', 'A-10', 'https://www.umassathletics.com'),  # Atlantic 10 D1
    ('uncw', 'CAA', 'https://www.uncwsports.com'),  # Colonial Athletic D1
    ('ut-arlington', 'WAC', 'https://www.utamavs.com'),  # Western Athletic D1
    ('utah-state', 'MWC', 'https://www.utahstateaggies.com'),  # Mountain West D1
    ('utep', 'C-USA', 'https://www.utepathletics.com'),  # Conference USA D1
    ('utrgv', 'WAC', 'https://www.utrgvvaqueros.com'),  # Western Athletic D1
    ('weber-state', 'Big Sky', 'https://www.weberstatesports.com'),  # Big Sky D1
    ('wyoming', 'MWC', 'https://www.wyomingathletics.com'),  # Mountain West D1
]

# Non-D1 teams (D2, D3, NAIA, etc.)
NON_D1_TEAMS = [
    ('anderson-in', 'Non-D1', None),  # Anderson University (IN) - D3
    ('augustana-college-il', 'Non-D1', None),  # Augustana College (IL) - D3
    ('aurora', 'Non-D1', None),  # Aurora University - D3
    ('baker', 'Non-D1', None),  # Baker University - NAIA
    ('barry', 'Non-D1', None),  # Barry University - D2
    ('bethany-ks-bethany', 'Non-D1', None),  # Bethany College (KS) - NAIA
    ('birmingham-southern', 'Non-D1', None),  # Birmingham-Southern - D3 (closed)
    ('cal-state-los-angeles', 'Non-D1', None),  # Cal State LA - D2
    ('centenary', 'Non-D1', None),  # Centenary College - D3
    ('central-missouri-state', 'Non-D1', None),  # Central Missouri - D2
    ('chicago-state', 'Non-D1', None),  # Chicago State - discontinued baseball
    ('coe-college', 'Non-D1', None),  # Coe College - D3
    ('college-of-charleston', 'CAA', 'https://www.cofcsports.com'),  # Actually D1 - CAA
    ('depauw', 'Non-D1', None),  # DePauw University - D3
    ('emporia-st', 'Non-D1', None),  # Emporia State - D2
    ('florida-southern', 'Non-D1', None),  # Florida Southern - D2
    ('fort-lauderdale', 'Non-D1', None),  # Unknown/Defunct
    ('grand-view', 'Non-D1', None),  # Grand View - NAIA
    ('hawaii-hilo', 'Non-D1', None),  # Hawaii-Hilo - D2
    ('lincoln', 'Non-D1', None),  # Lincoln University - D2
    ('linfield', 'Non-D1', None),  # Linfield University - D3
    ('loras-college', 'Non-D1', None),  # Loras College - D3
    ('mercyhurst', 'Non-D1', None),  # Mercyhurst - D2
    ('mid-america-christian', 'Non-D1', None),  # Mid-America Christian - NAIA
    ('nebraska-kearney', 'Non-D1', None),  # Nebraska-Kearney - D2
    ('north-georgia', 'Non-D1', None),  # North Georgia - D2
    ('northwestern-college-ia-northwestern-college', 'Non-D1', None),  # Northwestern (IA) - NAIA
    ('ny-institute-of-technology', 'Non-D1', None),  # NYIT - D2
    ('pacific-lutheran', 'Non-D1', None),  # Pacific Lutheran - D3
    ('southwest', 'Non-D1', None),  # Southwest University - NAIA
    ('st-gregory', 'Non-D1', None),  # St. Gregory - NAIA
    ('st-martins', 'Non-D1', None),  # St. Martin's - D2
    ('st-olaf', 'Non-D1', None),  # St. Olaf - D3
    ('st-thomas-minnesota', 'Non-D1', None),  # St. Thomas (MN) - recently moved to D1 but not sure about baseball
    ('sw-oklahoma-state', 'Non-D1', None),  # SW Oklahoma State - D2
    ('tabor', 'Non-D1', None),  # Tabor College - NAIA
    ('texas-lutheran', 'Non-D1', None),  # Texas Lutheran - D3
    ('texas-pan-american', 'Non-D1', None),  # Texas-Pan American - merged into UTRGV
    ('truman-state', 'Non-D1', None),  # Truman State - D2
    ('upper-iowa', 'Non-D1', None),  # Upper Iowa - D2
    ('wartburg', 'Non-D1', None),  # Wartburg - D3
    ('wayne-state-ne', 'Non-D1', None),  # Wayne State (NE) - D2
    ('west-georgia', 'Non-D1', None),  # West Georgia - D2
    ('william', 'Non-D1', None),  # William Woods - NAIA
    ('wisconsin-platteville', 'Non-D1', None),  # Wisconsin-Platteville - D3
]

def apply_updates():
    conn = sqlite3.connect(str(DB_PATH))
    
    all_updates = BATCH_UPDATES + [(team_id, conf, url) for team_id, conf, url in NON_D1_TEAMS]
    
    for team_id, conference, athletics_url in all_updates:
        if athletics_url:
            conn.execute("""
                UPDATE teams 
                SET conference = ?, athletics_url = ? 
                WHERE id = ?
            """, (conference, athletics_url, team_id))
        else:
            conn.execute("""
                UPDATE teams 
                SET conference = ? 
                WHERE id = ?
            """, (conference, team_id))
    
    conn.commit()
    conn.close()
    print(f"Updated {len(all_updates)} teams")

if __name__ == '__main__':
    apply_updates()