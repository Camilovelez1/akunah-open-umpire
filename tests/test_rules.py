from open_umpire.rules import Match, MatchConfig, Referee, Side

L, R = Side.LEFT, Side.RIGHT


def make(**cfg):
    match = Match(config=MatchConfig(**cfg))
    return match, Referee(match)


def test_serve_alternates_every_two_points():
    match, _ = make()
    servers = []
    for _ in range(6):
        servers.append(match.server)
        match.add_point(0)
    assert servers == [0, 0, 1, 1, 0, 0]


def test_deuce_alternates_every_point_and_needs_two_clear():
    match, _ = make()
    for _ in range(10):
        match.add_point(0)
        match.add_point(1)
    assert match.scores == [10, 10]
    first = match.server
    match.add_point(0)
    assert match.server != first
    assert match.games == [0, 0]
    match.add_point(0)
    assert match.games == [1, 0]
    assert match.scores == [0, 0]


def test_game_win_swaps_ends_and_first_server():
    match, _ = make()
    for _ in range(11):
        match.add_point(0)
    assert match.games == [1, 0]
    assert match.left_player == 1
    assert match.server == 1


def test_match_finishes_after_games_to_win():
    match, _ = make(games_to_win=2)
    for _ in range(22):
        match.add_point(1)
    assert match.finished
    assert match.games == [0, 2]


def test_undo_restores_previous_state():
    match, _ = make()
    for _ in range(11):
        match.add_point(0)
    match.undo()
    assert match.scores == [10, 0]
    assert match.games == [0, 0]
    assert match.left_player == 0


def test_good_rally_then_receiver_fails_to_return():
    match, ref = make()
    # Player 0 is on the left and serves.
    assert ref.bounce(L, 0.0) is None
    assert ref.bounce(R, 0.3) is None
    assert ref.bounce(L, 1.0) is None
    assert ref.bounce(R, 1.7) is None
    assert ref.tick(2.0) is None
    event = ref.tick(3.3)
    assert event.winner == 0
    assert match.scores == [1, 0]


def test_serve_bouncing_on_receiver_half_first_is_a_fault():
    match, ref = make()
    event = ref.bounce(R, 0.0)
    assert event.winner == 1


def test_double_bounce_ends_rally_immediately():
    match, ref = make()
    ref.bounce(L, 0.0)
    ref.bounce(R, 0.3)
    event = ref.bounce(R, 0.6)
    assert event.winner == 0


def test_serve_that_only_touches_own_half_loses():
    match, ref = make()
    ref.bounce(L, 0.0)
    event = ref.tick(2.0)
    assert event.winner == 1


def test_serve_that_never_bounces_loses():
    match, ref = make()
    ref.start_serve(0.0)
    event = ref.tick(2.0)
    assert event.winner == 1


def test_bounces_during_dead_time_are_ignored():
    match, ref = make(dead_time=2.0)
    ref.bounce(R, 0.0)  # fault, point to player 1 at t=0
    assert ref.bounce(R, 0.5) is None
    assert ref.bounce(R, 1.0) is None
    assert match.scores == [0, 1]
    assert not ref.in_rally


def test_server_on_right_after_swapping_ends():
    match, ref = make()
    match.swap_ends()  # player 0 now on the right, still serving
    assert ref.bounce(R, 0.0) is None
    assert ref.bounce(L, 0.3) is None
    event = ref.tick(5.0)
    assert event.winner == 0


def test_handicap_start_scores():
    match = Match(start_scores=(0, 4))
    assert match.scores == [0, 4]
    for _ in range(7):
        match.add_point(1)
    assert match.games == [0, 1]
    assert match.scores == [0, 4]
