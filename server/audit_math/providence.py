"""
Library for performing a Providence ballot polling risk-limiting audit,
as described by Broadrick et al https://arxiv.org/abs/2210.08717
"""

from __future__ import annotations

from collections import defaultdict
import math

from scipy.special import logsumexp
from scipy.stats import binom

from .sampler_contest import Contest
from .ballot_polling_types import (
    SampleSizeOption,
    BALLOT_POLLING_ROUND_SIZES,
    BALLOT_POLLING_SAMPLE_RESULTS,
)


def get_sample_size(
    risk_limit: int,
    contest: Contest,
    sample_results: BALLOT_POLLING_SAMPLE_RESULTS | None,
    round_sizes: BALLOT_POLLING_ROUND_SIZES | None,
) -> dict[str, SampleSizeOption]:
    """
    Computes sample sizes for the next round, parameterized by likelihood that the
    sample will confirm the election result, assuming accurate results.
    For each target stopping probability in (0.7, 0.8, 0.9) does the following.
    For each unfinished winner-loser pair, does a binary search for pairwise round
    sizes that meet the target stopping probability. Then scales each pair's round
    size to whole-contest draws and takes the largest across pairs.

    Inputs:
        risk_limit:     maximum risk as an integer percentage
        contest:        a sampler_contest object of the contest being audited
        sample_results: map round ids to mapping of candidates to incremental votes
        round_sizes:    map round nums to tuples of round ids and incremental round sizes

    Outputs:
        samples:        dictionary mapping confirmation likelihood to additional
                        whole-contest draws for the next round
    """
    alpha = risk_limit / 100
    if not 0 < alpha < 1:
        raise ValueError("The risk-limit must be greater than zero and less than 100.")
    if bool(sample_results) != bool(round_sizes):
        raise ValueError(
            "Sample results and round sizes must both describe completed rounds."
        )

    contest = pad_zero_vote_totals(contest)
    if not contest.losers:
        raise ValueError("Contest must have candidates who did not win!")

    probabilities = (0.7, 0.8, 0.9)
    samples: dict[str, SampleSizeOption] = {
        str(prob): {"type": None, "size": 0, "prob": prob} for prob in probabilities
    }
    completed_rounds = sorted((round_sizes or {}).items())

    for winner, winner_votes in contest.winners.items():
        for loser, loser_votes in contest.losers.items():
            if winner_votes <= loser_votes:
                raise ValueError(
                    "Each reported winner must have more votes than each loser."
                )
            winner_fraction = winner_votes / (winner_votes + loser_votes)
            previous_winner_votes = previous_loser_votes = 0
            pair_passed = False

            for round_index, (_, round_info) in enumerate(completed_rounds):
                if round_index > 0 and round_info.round_size <= 0:
                    raise ValueError(
                        "Invalid sample size, must be larger than previous round."
                    )
                round_votes = (sample_results or {})[round_info.round_id]
                current_winner_votes = round_votes.get(winner, 0)
                current_loser_votes = round_votes.get(loser, 0)
                if (
                    pairwise_compute_risk(
                        winner_fraction,
                        current_winner_votes,
                        current_loser_votes,
                        previous_winner_votes,
                        previous_loser_votes,
                    )
                    <= alpha
                ):
                    pair_passed = True
                    break
                previous_winner_votes += current_winner_votes
                previous_loser_votes += current_loser_votes

            if pair_passed:
                continue

            # Estimate whole-contest draws from relevant pair votes. Padding must
            # not make this conversion reduce the required number of draws.
            ballots_per_pair_vote = max(
                1, contest.ballots / (winner_votes + loser_votes)
            )
            for prob in probabilities:
                pair_size = pairwise_get_sample_size(
                    alpha,
                    winner_fraction,
                    prob,
                    previous_winner_votes,
                    previous_loser_votes,
                )
                samples[str(prob)]["size"] = max(
                    samples[str(prob)]["size"],
                    math.ceil(pair_size * ballots_per_pair_vote),
                )

    # As in R2B2, dilution and the maximum across pairs are planning estimates,
    # not an exact joint probability that every pair will pass in the next round.
    return samples


def compute_risk(
    risk_limit: int,
    contest: Contest,
    sample_results: BALLOT_POLLING_SAMPLE_RESULTS,
    round_sizes: BALLOT_POLLING_ROUND_SIZES,
) -> tuple[dict[tuple[str, str], float], bool]:
    """
    Computes the risk-value of <sample_results> based on results in <contest>.

    Preserve the existing aggregation: stop evaluating a pair once it passes,
    and report the minimum across rounds of the maximum active pairwise risk.

    Inputs:
        risk_limit:     maximum risk as an integer percentage
        contest:        a sampler_contest object of the contest being measured
        sample_results: map round ids to mapping of candidates to incremental votes
        round_sizes:    map round nums to tuples of round ids and incremental round sizes

    Outputs:
        measurements:   the p-value of the hypotheses that the election
                        result is correct based on the sample
        confirmed:      a boolean indicating whether the audit can stop
    """
    alpha = risk_limit / 100
    if not 0 < alpha < 1:
        raise ValueError("The risk-limit must be greater than zero and less than 100.")
    if not sample_results or not round_sizes:
        raise ValueError("At least one completed round is required to compute risk.")

    contest = pad_zero_vote_totals(contest)

    pairs = {
        (winner, loser): winner_votes / (winner_votes + loser_votes)
        for winner, winner_votes in contest.winners.items()
        for loser, loser_votes in contest.losers.items()
        if winner_votes > loser_votes
    }
    passed_pairs: set[tuple[str, str]] = set()
    previous_votes: dict[str, int] = defaultdict(int)
    risk = math.inf

    for round_index, (_, round_info) in enumerate(sorted(round_sizes.items())):
        if round_index > 0 and round_info.round_size <= 0:
            raise ValueError("Invalid sample size, must be larger than previous round.")
        round_votes = sample_results[round_info.round_id]
        round_risk = 0.0
        for (winner, loser), winner_fraction in pairs.items():
            if (winner, loser) in passed_pairs:
                continue
            pair_risk = pairwise_compute_risk(
                winner_fraction,
                round_votes.get(winner, 0),
                round_votes.get(loser, 0),
                previous_votes[winner],
                previous_votes[loser],
            )
            round_risk = max(round_risk, pair_risk)
            if pair_risk <= alpha:
                passed_pairs.add((winner, loser))
        risk = min(risk, round_risk)
        if len(passed_pairs) == len(pairs):
            break
        for candidate, votes in round_votes.items():
            previous_votes[candidate] += votes

    return {("winner", "loser"): risk}, risk <= alpha


def pairwise_get_sample_size(
    alpha: float,
    winner_fraction: float,
    stopping_probability: float,
    previous_winner_votes: int = 0,
    previous_loser_votes: int = 0,
) -> int:
    """Estimate additional pair votes needed for the requested 
    stopping probability by binary search.
    """
    if not 0 < alpha < 1 or not 0 < stopping_probability < 1:
        raise ValueError("The risk limit and stopping probability must be in (0, 1).")

    def stopping_prob(round_size: int) -> float:
        kmin = pairwise_find_kmin(
            alpha,
            winner_fraction,
            round_size,
            previous_winner_votes,
            previous_loser_votes,
        )
        if kmin is None:
            return 0.0
        return float(binom.sf(kmin - 1, round_size, winner_fraction))

    left, right = 0, 1
    while stopping_prob(right) < stopping_probability:
        left, right = right, 2 * right

    # Keep right as a size verified to meet the requested probability.
    while left + 1 < right:
        mid = (left + right) // 2
        if stopping_prob(mid) >= stopping_probability:
            right = mid
        else:
            left = mid
    return right


def pairwise_find_kmin(
    alpha: float,
    winner_fraction: float,
    round_size: int,
    previous_winner_votes: int = 0,
    previous_loser_votes: int = 0,
) -> int | None:
    """Find the minimum passing winner count for a round of additional pair votes.

    previous_* counts are cumulative over preceding rounds. Return None if no
    winner count can pass at this round size; zero is a valid passing count.
    """
    if not 0 < alpha < 1:
        raise ValueError("The risk limit must be in (0, 1).")
    if not 0.5 < winner_fraction < 1:
        raise ValueError("The reported winner fraction must be in (0.5, 1).")
    if round_size < 0:
        raise ValueError("The round size must be nonnegative.")
    if min(previous_winner_votes, previous_loser_votes) < 0:
        raise ValueError("Sample vote counts must be nonnegative.")

    log_winner_ratio = math.log(2 * winner_fraction)
    log_loser_ratio = math.log(2 * (1 - winner_fraction))
    log_prior_ratio = -(
        previous_winner_votes * log_winner_ratio
        + previous_loser_votes * log_loser_ratio
    )

    def passes(winner_votes: int) -> bool:
        return (
            pairwise_compute_risk(
                winner_fraction,
                winner_votes,
                round_size - winner_votes,
                previous_winner_votes,
                previous_loser_votes,
            )
            <= alpha
        )

    # BRAVO's passing winner count bounds Providence's from above.
    bravo_kmin = math.ceil(
        (log_prior_ratio - math.log(alpha) - round_size * log_loser_ratio)
        / (log_winner_ratio - log_loser_ratio)
    )
    left, right = 0, min(round_size, max(0, bravo_kmin))
    if not passes(right):
        # Verify the bound rather than relying on floating-point rounding.
        if right == round_size or not passes(round_size):
            return None
        right = round_size

    # For a fixed round size, risk decreases with the winner vote count.
    while left < right:
        mid = (left + right) // 2
        if passes(mid):
            right = mid
        else:
            left = mid + 1
    return left


def pairwise_compute_risk(
    winner_fraction: float,
    winner_votes: int,
    loser_votes: int,
    previous_winner_votes: int = 0,
    previous_loser_votes: int = 0,
) -> float:
    """Return the reciprocal Providence statistic capped at 1.

    winner_fraction is the reported winner's reported fraction of this pair's votes.
    winner_votes and loser_votes are counts for the current round;
    previous_* counts are cumulative counts over all preceding rounds for this pair.

    The reciprocal statistic is the prior sample's null/reported likelihood
    ratio multiplied by the current round's null/reported binomial tail ratio.
    """
    if not 0 < winner_fraction < 1:
        # Disallow winner_fraction=1 since then the alternative binomial pmf gives 
        # zero probability to any number of votes for the reported loser, giving a  
        # test statistic which can never thereafter meet the stopping condition.
        # (Note that compute_risk function above always pads vote counts 
        # to >=1 for all candidates, avoiding this situation.)
        raise ValueError("The reported winner fraction must be in (0, 1).")
    if min(winner_votes, loser_votes, previous_winner_votes, previous_loser_votes) < 0:
        raise ValueError("Sample vote counts must be nonnegative.")

    round_size = winner_votes + loser_votes
    tail_counts = range(winner_votes, round_size + 1)

    # Likelihood ratio for previous rounds on logscale. 
    log_prior_ratio = -(
        previous_winner_votes * math.log(2 * winner_fraction)
        + previous_loser_votes * math.log(2 * (1 - winner_fraction))
    )
    # Current round's tails on logscale. 
    log_tail_ratio = (
        logsumexp(binom.logpmf(tail_counts, round_size, 0.5))
        - logsumexp(binom.logpmf(tail_counts, round_size, winner_fraction))
    )
    log_risk = log_prior_ratio + log_tail_ratio
    # Match BRAVO's cap at 1, checking before exponentiating to avoid overflow.
    if log_risk >= 0:
        return 1.0
    return math.exp(log_risk)


def pad_zero_vote_totals(contest: Contest) -> Contest:
    """Pad zero reported totals without modifying the input contest."""
    # For any candidate with zero reported votes, instead assume they have one vote.
    # This avoids degenerate cases and preserves the risk-limiting property.
    if 0 in contest.candidates.values():
        contest = Contest(
            contest.name,
            {
                **{
                    candidate: 1 if votes == 0 else votes
                    for candidate, votes in contest.candidates.items()
                },
                "ballots": contest.ballots,
                "numWinners": contest.num_winners,
                "votesAllowed": contest.votes_allowed,
                "pendingBallots": contest.pending_ballots,
                "isSubjectToRunoff": contest.is_subject_to_runoff,
            },
        )
    return contest
