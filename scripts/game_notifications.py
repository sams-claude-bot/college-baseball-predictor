#!/usr/bin/env python3
"""
Shared game notification dispatcher.

Extracted from statbroadcast_poller._check_notifications so that any poller
(StatBroadcast, SIDEARM, ESPN Fastcast) can trigger the same push notification
logic with a consistent interface.

Usage:
    dispatcher = GameNotificationDispatcher(conn)
    dispatcher.check(game_id, situation_dict)
    dispatcher.check_final(game_id)

The situation dict should include at minimum:
    - inning: int
    - inning_half: 'top' | 'bottom'
    - home_score: int
    - visitor_score: int

Optional fields (for upset watch):
    - outs: int
    - on_first: bool
    - on_second: bool
    - on_third: bool
"""

import logging
import sqlite3
from typing import Optional

logger = logging.getLogger('game_notifications')


def _to_int(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _norm_half(value) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v.startswith('top') or v == 't':
        return 'top'
    if v.startswith('bot') or v.startswith('bottom') or v == 'b':
        return 'bottom'
    return v or None


def _ordinal(n) -> str:
    n = _to_int(n)
    if n is None:
        return '?'
    if 10 <= (n % 100) <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def _half_label(inning_num, half_val) -> str:
    h = _norm_half(half_val)
    prefix = 'Top' if h == 'top' else 'Bot'
    return f"{prefix} {_ordinal(inning_num)}"


def _get_game_teams(conn: sqlite3.Connection, game_id: str):
    """Fetch team info for a game. Returns dict or None."""
    row = conn.execute(
        """
        SELECT g.home_team_id, g.away_team_id,
               h.name as home_name, a.name as away_name,
               h.conference as home_conf, a.conference as away_conf
        FROM games g
        JOIN teams h ON g.home_team_id = h.id
        JOIN teams a ON g.away_team_id = a.id
        WHERE g.id = ?
        """,
        (game_id,),
    ).fetchone()
    if not row:
        return None

    if isinstance(row, tuple):
        return {
            'home_team_id': row[0], 'away_team_id': row[1],
            'home_name': row[2], 'away_name': row[3],
            'home_conf': row[4], 'away_conf': row[5],
        }
    return {
        'home_team_id': row['home_team_id'], 'away_team_id': row['away_team_id'],
        'home_name': row['home_name'], 'away_name': row['away_name'],
        'home_conf': row['home_conf'], 'away_conf': row['away_conf'],
    }


class GameNotificationDispatcher:
    """Stateless-per-game notification dispatcher.

    Tracks per-game half-inning and score state to detect transitions,
    then fires the appropriate push notifications.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._last_half_state: dict = {}      # game_id → (inning, half)
        self._half_start_score: dict = {}     # game_id → (visitor, home)
        self._last_score_state: dict = {}     # game_id → (visitor, home)

    def check(self, game_id: str, situation: dict):
        """Check game state for notification triggers.

        Triggers:
        1. Half-inning transition (all) → game_update
        2. Scoring half-inning recap → game_update_scoring
        3. Instant score change → score_change
        4. SEC upset watch (on half transitions) → upset_watch
        """
        try:
            inning = _to_int(situation.get('inning'))
            half = _norm_half(situation.get('inning_half'))
            home_score = _to_int(situation.get('home_score'))
            visitor_score = _to_int(situation.get('visitor_score'))

            if inning is None or half is None or home_score is None or visitor_score is None:
                return

            current_half_state = (inning, half)
            current_score = (visitor_score, home_score)

            prev_half_state = self._last_half_state.get(game_id)
            prev_score_state = self._last_score_state.get(game_id)

            # First sighting: seed state to avoid startup spam.
            if prev_half_state is None or prev_score_state is None:
                self._last_half_state[game_id] = current_half_state
                self._last_score_state[game_id] = current_score
                self._half_start_score[game_id] = current_score
                return

            half_transition = current_half_state != prev_half_state

            completed_half_label = None
            scored_in_completed_half = False
            half_delta_away = 0
            half_delta_home = 0
            if half_transition:
                start_score = self._half_start_score.get(game_id, prev_score_state)
                start_away = _to_int(start_score[0]) or 0
                start_home = _to_int(start_score[1]) or 0

                half_delta_away = current_score[0] - start_away
                half_delta_home = current_score[1] - start_home
                scored_in_completed_half = half_delta_away > 0 or half_delta_home > 0
                completed_half_label = _half_label(prev_half_state[0], prev_half_state[1])

            score_changed = current_score != prev_score_state
            away_delta = current_score[0] - prev_score_state[0]
            home_delta = current_score[1] - prev_score_state[1]
            runs_scored_now = away_delta > 0 or home_delta > 0

            # Update state caches before any early returns.
            self._last_half_state[game_id] = current_half_state
            self._last_score_state[game_id] = current_score
            if half_transition:
                self._half_start_score[game_id] = current_score
            elif game_id not in self._half_start_score:
                self._half_start_score[game_id] = current_score

            if not half_transition and not (score_changed and runs_scored_now):
                return

            teams = _get_game_teams(self.conn, game_id)
            if not teams:
                return

            home_tid = teams['home_team_id']
            away_tid = teams['away_team_id']
            home_name = teams['home_name']
            away_name = teams['away_name']
            home_conf = teams['home_conf']
            away_conf = teams['away_conf']

            inning_label = situation.get('inning_display', _half_label(inning, half))

            from notifications import (
                send_team_notification, send_game_notification,
                send_conference_notification, ensure_tables,
            )
            ensure_tables(self.conn)

            score_line = f"{away_name} {visitor_score}, {home_name} {home_score}"

            # --- 1. Legacy half-inning updates (all transitions) ---
            if half_transition:
                for team_id in (home_tid, away_tid):
                    dedup = f"game_update:{game_id}:{team_id}:{inning}:{half}"
                    send_team_notification(
                        team_id,
                        'game_update',
                        {
                            'title': f"⚾ {score_line}",
                            'body': inning_label,
                            'url': f"/game/{game_id}",
                            'tag': f"game-{game_id}",
                            'game_id': game_id,
                        },
                        dedup_key=dedup,
                        conn=self.conn,
                    )

            # --- 2. Half-inning scoring recaps only ---
            if half_transition and scored_in_completed_half:
                delta_bits = []
                if half_delta_away > 0:
                    delta_bits.append(f"{away_name} +{half_delta_away}")
                if half_delta_home > 0:
                    delta_bits.append(f"{home_name} +{half_delta_home}")
                delta_text = ', '.join(delta_bits) if delta_bits else 'Runs scored'

                recap_payload = {
                    'title': f"📌 Inning Recap: {score_line}",
                    'body': f"{completed_half_label} • {delta_text}",
                    'url': f"/game/{game_id}",
                    'tag': f"inning-score-{game_id}",
                    'game_id': game_id,
                }

                for team_id in (home_tid, away_tid):
                    dedup = (
                        f"game_update_scoring:{game_id}:{team_id}:"
                        f"{prev_half_state[0]}:{prev_half_state[1]}:{visitor_score}-{home_score}"
                    )
                    send_team_notification(
                        team_id,
                        'game_update_scoring',
                        recap_payload,
                        dedup_key=dedup,
                        conn=self.conn,
                    )

                # Also deliver to explicit game-follow subscribers.
                send_game_notification(
                    game_id,
                    'game_update_scoring',
                    recap_payload,
                    dedup_key=(
                        f"game_update_scoring:{game_id}:game:"
                        f"{prev_half_state[0]}:{prev_half_state[1]}:{visitor_score}-{home_score}"
                    ),
                    conn=self.conn,
                )

            # --- 3. Instant scoring alerts ---
            if score_changed and runs_scored_now:
                delta_bits = []
                if away_delta > 0:
                    delta_bits.append(f"{away_name} +{away_delta}")
                if home_delta > 0:
                    delta_bits.append(f"{home_name} +{home_delta}")
                delta_text = ', '.join(delta_bits) if delta_bits else 'Score changed'

                score_payload = {
                    'title': f"🚨 Score Change: {score_line}",
                    'body': f"{delta_text} • {inning_label}",
                    'url': f"/game/{game_id}",
                    'tag': f"score-change-{game_id}",
                    'game_id': game_id,
                }

                for team_id in (home_tid, away_tid):
                    dedup = f"score_change:{game_id}:{team_id}:{visitor_score}-{home_score}"
                    send_team_notification(
                        team_id,
                        'score_change',
                        score_payload,
                        dedup_key=dedup,
                        conn=self.conn,
                    )

                send_game_notification(
                    game_id,
                    'score_change',
                    score_payload,
                    dedup_key=f"score_change:{game_id}:game:{visitor_score}-{home_score}",
                    conn=self.conn,
                )

            # --- 4. SEC Upset Watch (only on half transitions) ---
            if half_transition:
                sec_team = None
                if home_conf == 'SEC' and away_conf != 'SEC':
                    sec_team = 'home'
                elif away_conf == 'SEC' and home_conf != 'SEC':
                    sec_team = 'away'

                if sec_team:
                    try:
                        from models.win_probability import WinProbabilityModel
                        wp_model = WinProbabilityModel()
                        home_wp = wp_model.calculate(
                            home_score=int(home_score),
                            away_score=int(visitor_score),
                            inning=int(inning),
                            inning_half=half,
                            outs=int(situation.get('outs', 0)),
                            on_first=situation.get('on_first', False),
                            on_second=situation.get('on_second', False),
                            on_third=situation.get('on_third', False),
                            game_id=game_id,
                        )

                        sec_losing = (sec_team == 'home' and home_wp < 0.25) or \
                                     (sec_team == 'away' and home_wp > 0.75)

                        if sec_losing:
                            sec_name = home_name if sec_team == 'home' else away_name
                            opp_name = away_name if sec_team == 'home' else home_name
                            sec_wp = home_wp if sec_team == 'home' else (1 - home_wp)
                            lose_pct = (1 - sec_wp) * 100

                            send_conference_notification(
                                'SEC',
                                'upset_watch',
                                {
                                    'title': f"⚠️ SEC Upset Watch: {sec_name}",
                                    'body': (
                                        f"{sec_name} has {lose_pct:.0f}% chance to lose vs "
                                        f"{opp_name} | {score_line} ({inning_label})"
                                    ),
                                    'url': f"/game/{game_id}",
                                    'tag': f"upset-{game_id}",
                                    'game_id': game_id,
                                },
                                dedup_key=f"upset:{game_id}",
                                conn=self.conn,
                            )
                    except Exception as e:
                        logger.warning("WP calculation failed for upset check: %s", e)

        except Exception as e:
            logger.warning("Notification check error for %s: %s", game_id, e, exc_info=True)

    def check_final(self, game_id: str):
        """Send final score notifications when a game completes.

        Sends to both team-follow AND game-follow subscribers.
        """
        try:
            from notifications import (
                send_team_notification, send_game_notification, ensure_tables,
            )
            ensure_tables(self.conn)

            row = self.conn.execute(
                """
                SELECT g.home_team_id, g.away_team_id,
                       h.name as home_name, a.name as away_name,
                       g.home_score, g.away_score, g.innings
                FROM games g
                JOIN teams h ON g.home_team_id = h.id
                JOIN teams a ON g.away_team_id = a.id
                WHERE g.id = ?
                """,
                (game_id,),
            ).fetchone()

            if not row:
                return

            if isinstance(row, tuple):
                home_tid, away_tid = row[0], row[1]
                home_name, away_name = row[2], row[3]
                h_score, a_score = row[4], row[5]
                innings = row[6]
            else:
                home_tid = row['home_team_id']
                away_tid = row['away_team_id']
                home_name = row['home_name']
                away_name = row['away_name']
                h_score = row['home_score']
                a_score = row['away_score']
                innings = row['innings']

            extra = f" ({innings})" if innings and innings > 9 else ""
            winner = home_name if h_score > a_score else away_name
            score_line = f"{away_name} {a_score}, {home_name} {h_score}"

            final_payload = {
                'title': f"🏁 Final{extra}: {score_line}",
                'body': f"{winner} wins!",
                'url': f"/game/{game_id}",
                'tag': f"final-{game_id}",
                'game_id': game_id,
                'removeGame': True,
            }

            # Send to team-follow subscribers.
            for team_id in (home_tid, away_tid):
                send_team_notification(
                    team_id,
                    'final_score',
                    final_payload,
                    dedup_key=f"final:{game_id}:{team_id}",
                    conn=self.conn,
                )

            # Send to game-follow subscribers (fixes missing delivery).
            send_game_notification(
                game_id,
                'final_score',
                final_payload,
                dedup_key=f"final:{game_id}:game",
                conn=self.conn,
            )

            # Remove completed game from followed-game preferences and
            # account favorites so it drops off the user's list.
            self._cleanup_finished_game(game_id)

        except Exception as e:
            logger.warning("Final notification error for %s: %s", game_id, e)

    def _cleanup_finished_game(self, game_id: str):
        """Remove a finished game from all followed-game lists (server-side).

        Deletes:
        - alert_preferences rows where game_id matches (push notification prefs)
        - account_favorite_games rows where game_id matches (light account sync)
        """
        try:
            self.conn.execute(
                "DELETE FROM alert_preferences WHERE game_id = ?",
                (game_id,),
            )
        except Exception:
            pass  # Table may not exist

        try:
            self.conn.execute(
                "DELETE FROM account_favorite_games WHERE game_id = ?",
                (game_id,),
            )
        except Exception:
            pass  # Table may not exist

        try:
            self.conn.commit()
        except Exception:
            pass

        logger.debug("Cleaned up finished game %s from followed lists", game_id)

    def cleanup_game(self, game_id: str):
        """Remove cached state for a completed game."""
        self._last_half_state.pop(game_id, None)
        self._half_start_score.pop(game_id, None)
        self._last_score_state.pop(game_id, None)
