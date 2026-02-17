"""
D1Baseball slug mappings — DB team_id to D1Baseball URL slug.
Only includes teams where the slug differs from the DB id.
If a team_id is not in this dict, use the team_id directly as the slug.
"""

DB_TO_D1_SLUG = {
    # SEC
    'ole-miss': 'olemiss',
    'mississippi-state': 'mississippistate',
    'texas-am': 'texasam',
    'south-carolina': 'southcarolina',
    # ACC
    'boston-college': 'bostoncollege',
    'florida-state': 'floridastate',
    'miami-fl': 'miami',
    'nc-state': 'ncstate',
    'north-carolina': 'northcarolina',
    'notre-dame': 'notredame',
    'virginia-tech': 'virginiatech',
    'wake-forest': 'wakeforest',
    # Big 12
    'arizona-state': 'arizonastate',
    'iowa-state': 'iowastate',
    'kansas-state': 'kansasstate',
    'oklahoma-state': 'oklahomastate',
    'texas-tech': 'texastech',
    'west-virginia': 'westvirginia',
    # Big Ten
    'michigan-state': 'michiganstate',
    'ohio-state': 'ohiostate',
    'penn-state': 'pennstate',
    # Georgia Tech already handled
    'georgia-tech': 'gatech',
    'vanderbilt': 'vandy',
}

def get_d1_slug(team_id):
    """Get D1Baseball URL slug for a team_id."""
    return DB_TO_D1_SLUG.get(team_id, team_id)
