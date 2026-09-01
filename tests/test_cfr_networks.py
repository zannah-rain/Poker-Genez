import os
import tempfile

import numpy as np
import pytest
import torch

import strategy
from cfr_networks import (
    AdvantageNet, AdvantageNetConfig, _normalized_mean_abs_shap, _validity_codes, clone, interaction_strength_for_feature,
    load, mean_shap_contributions_for_samples, pairwise_interaction_strength, save,
)
from cfr_reservoir import ReservoirBuffer
from features import MASKED


class TestAdvantageNetForward:
    def test_output_shape_matches_num_action_categories(self):
        net = AdvantageNet(input_dim=5, hidden_sizes=(8, 8))
        x = torch.zeros((3, 5))
        out = net(x)
        assert out.shape == (3, strategy.NUM_ACTION_CATEGORIES)

    def test_predict_returns_1d_numpy_array(self):
        net = AdvantageNet(input_dim=4, hidden_sizes=(8,))
        features = np.zeros(4, dtype=np.float32)
        out = net.predict(features)
        assert isinstance(out, np.ndarray)
        assert out.shape == (strategy.NUM_ACTION_CATEGORIES,)

    def test_zero_hidden_layers_is_a_linear_model(self):
        net = AdvantageNet(input_dim=3, hidden_sizes=())
        x = torch.zeros((1, 3))
        out = net(x)
        assert out.shape == (1, strategy.NUM_ACTION_CATEGORIES)


class TestSaveLoadRoundTrip:
    def test_reloaded_net_produces_identical_predictions(self):
        net = AdvantageNet(input_dim=6, hidden_sizes=(16, 16))
        config = AdvantageNetConfig(
            feature_keys=("hand_category_norm", "street_norm", "is_aggressor_previous_street", "raises_preflop_norm", "call_amount_norm", "spr_norm"),
            hidden_sizes=(16, 16),
            table_size=6,
        )
        features = np.random.default_rng(0).random(6).astype(np.float32)
        original_prediction = net.predict(features)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "checkpoint")
            save(net, config, path)
            assert os.path.exists(f"{path}.pt")
            assert os.path.exists(f"{path}.json")

            loaded_net, loaded_config = load(path)
            reloaded_prediction = loaded_net.predict(features)

        assert loaded_config.feature_keys == config.feature_keys
        assert loaded_config.hidden_sizes == config.hidden_sizes
        assert loaded_config.table_size == config.table_size
        assert np.allclose(original_prediction, reloaded_prediction)

    def test_loaded_input_dim_matches_feature_key_count(self):
        keys = ("hand_category_norm", "street_norm")
        net = AdvantageNet(input_dim=len(keys), hidden_sizes=(8,))
        config = AdvantageNetConfig(feature_keys=keys, hidden_sizes=(8,), table_size=2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "checkpoint")
            save(net, config, path)
            loaded_net, _ = load(path)
        assert loaded_net.input_dim == 2


class TestClone:
    def test_clone_produces_identical_predictions(self):
        net = AdvantageNet(input_dim=5, hidden_sizes=(8, 8))
        features = np.random.default_rng(0).random(5).astype(np.float32)
        cloned = clone(net)
        assert np.allclose(net.predict(features), cloned.predict(features))

    def test_clone_is_independent_of_later_training_on_the_original(self):
        net = AdvantageNet(input_dim=4, hidden_sizes=(8,))
        features = np.random.default_rng(0).random(4).astype(np.float32)
        cloned = clone(net)
        before = cloned.predict(features)

        optimizer = torch.optim.SGD(net.parameters(), lr=1.0)
        net.train()
        loss = net(torch.from_numpy(features).unsqueeze(0)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        after = cloned.predict(features)
        assert np.allclose(before, after)  # unaffected by training on the original
        assert not np.allclose(net.predict(features), cloned.predict(features))  # original actually did change


def _filled_reservoir(capacity, feature_dim, rng):
    buf = ReservoirBuffer(capacity=capacity, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng)
    for _ in range(capacity):
        buf.add(
            rng.random(feature_dim).astype(np.float32),
            rng.random(strategy.NUM_ACTION_CATEGORIES).astype(np.float32),
            rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.5,
            float(rng.integers(1, 10)),
        )
    return buf


class TestNormalizedMeanAbsShap:
    def test_feature_constant_across_samples_scores_zero_despite_large_raw_value(self):
        # 2 actions, 5 samples, 3 features -- feature 1 contributes a
        # constant +20 to every sample/action, exactly the "looks important
        # but is actually uninformative for this pool" case.
        shap_values = [
            np.array([[1.0, 20.0, -3.0], [2.0, 20.0, 3.0], [-1.0, 20.0, 0.5], [0.5, 20.0, -2.0], [-2.0, 20.0, 1.0]]),
            np.array([[0.2, 20.0, 1.0], [-0.3, 20.0, -1.0], [0.1, 20.0, 2.0], [-0.2, 20.0, 0.0], [0.4, 20.0, -0.5]]),
        ]

        mean_abs = _normalized_mean_abs_shap(shap_values)

        assert mean_abs[1] == pytest.approx(0.0, abs=1e-9)
        assert mean_abs[0] > 0.0
        assert mean_abs[2] > 0.0

    def test_unnormalized_would_have_ranked_the_constant_feature_highest(self):
        # Same data as above: sanity-check that this scenario really would
        # fool a plain mean(|raw shap|) metric, motivating the normalization.
        shap_values = [
            np.array([[1.0, 20.0, -3.0], [2.0, 20.0, 3.0], [-1.0, 20.0, 0.5], [0.5, 20.0, -2.0], [-2.0, 20.0, 1.0]]),
            np.array([[0.2, 20.0, 1.0], [-0.3, 20.0, -1.0], [0.1, 20.0, 2.0], [-0.2, 20.0, 0.0], [0.4, 20.0, -0.5]]),
        ]
        raw_mean_abs = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=(0, 1))
        assert np.argmax(raw_mean_abs) == 1

        mean_abs = _normalized_mean_abs_shap(shap_values)
        assert np.argmax(mean_abs) != 1

    def test_varying_feature_unaffected_when_its_own_mean_is_zero(self):
        shap_values = [np.array([[1.0], [-1.0], [2.0], [-2.0]])]
        mean_abs = _normalized_mean_abs_shap(shap_values)
        assert mean_abs[0] == pytest.approx(1.5)  # mean(|1|,|1|,|2|,|2|) unchanged by centering on a zero mean


class TestValidityCodes:
    def test_identical_rows_get_the_same_code(self):
        valid = np.array([[True, False, True], [True, False, True]])
        codes = _validity_codes(valid)
        assert codes[0] == codes[1]

    def test_different_patterns_get_different_codes(self):
        valid = np.array([[True, True], [True, False], [False, True], [False, False]])
        codes = _validity_codes(valid)
        assert len(set(codes.tolist())) == 4

    def test_all_true_row_matches_across_widths_conceptually(self):
        # Not a real invariant of the encoding itself -- just documents that
        # an all-True row (nothing masked) always decodes to the same
        # "everything real" code regardless of which columns happen to be
        # maskable at all, since every unmaskable column contributes a
        # constant True that never differentiates rows.
        valid = np.array([[True, True, True]])
        assert _validity_codes(valid)[0] == (1 << 3) - 1


class TestMeanShapContributionsForSamples:
    # Real (non-maskable) features.FEATURE_NAMES keys, not placeholders --
    # mean_shap_contributions_for_samples now looks each key up in
    # features.py's catalog (see cfr_features.unmasked_validity) to know
    # whether it needs masked-row-aware stratification.
    _KEYS = ("hand_category_norm", "street_norm", "spr_norm", "raises_preflop_norm")

    def test_empty_pool_returns_empty_list(self):
        rng = np.random.default_rng(0)
        net = AdvantageNet(input_dim=4, hidden_sizes=(8,))
        empty = np.zeros((0, 4), dtype=np.float32)
        assert mean_shap_contributions_for_samples(
            net, empty, empty, self._KEYS, rng, sample_size=5, background_size=2, nsamples=5,
        ) == []

    def test_returns_one_entry_per_feature_sorted_descending_and_nonnegative(self):
        rng = np.random.default_rng(0)
        net = AdvantageNet(input_dim=4, hidden_sizes=(8,))
        features = _filled_reservoir(capacity=30, feature_dim=4, rng=rng).features

        result = mean_shap_contributions_for_samples(
            net, features, features, self._KEYS, rng, sample_size=10, background_size=5, nsamples=5,
        )

        assert sorted(k for k, _ in result) == sorted(self._KEYS)
        values = [v for _, v in result]
        assert all(v >= 0.0 for v in values)
        assert values == sorted(values, reverse=True)

    def test_a_feature_the_net_structurally_ignores_ranks_last_near_zero(self):
        rng = np.random.default_rng(1)
        net = AdvantageNet(input_dim=4, hidden_sizes=(8,))
        with torch.no_grad():
            first_layer = net.hidden[0].block[0]
            first_layer.weight[:, 2] = 0.0  # feature index 2 (spr_norm) can never affect any hidden unit
        features = _filled_reservoir(capacity=30, feature_dim=4, rng=rng).features

        result = mean_shap_contributions_for_samples(
            net, features, features, self._KEYS, rng, sample_size=10, background_size=5, nsamples=5,
        )

        by_key = dict(result)
        assert by_key["spr_norm"] == pytest.approx(0.0, abs=1e-6)
        assert result[-1][0] == "spr_norm"


class TestMeanShapContributionsForSamplesMaskedFeatures:
    """Regression coverage for a real bug: a `maskable` feature (see
    features.FeatureSpec) that's pure noise -- no real relationship to the
    net's output at all -- could still read as highly important under the
    default (unfiltered, mixed-street) SHAP computation, purely because it
    shares its masking condition (e.g. every draw-shape feature masks
    together at the river) with some other, genuinely important feature.
    A gradient-based explainer comparing a masked row against an unmasked
    background (or vice versa) sees a huge, off-training-domain jump in
    that one dimension and can misattribute the *other* feature's real
    effect onto it. mean_shap_contributions_for_samples now computes each
    masking-pattern stratum (see _validity_codes) separately, with
    background restricted to the same pattern, specifically to prevent
    this."""

    def test_masked_noise_feature_does_not_inflate_above_an_honest_control(self):
        rng = np.random.default_rng(0)
        n = 4000
        is_river = rng.random(n) < 0.25
        # draw_norm/hole_hand_grid_x_norm: pure noise, masked together at
        # "river" -- zero true relationship to the target, whether masked
        # or not (any two real `maskable` features sharing that same
        # masking condition would do equally well here; these two just
        # stand in for that). street_norm: the *real* cause of a
        # river-specific shift (standing in for the actual reason draw
        # features mask together with it). hole_suited: an honest,
        # never-masked, always-irrelevant control -- the noise floor these
        # should NOT exceed.
        keys = ("draw_norm", "hole_hand_grid_x_norm", "street_norm", "hole_suited")
        noise1 = np.where(is_river, MASKED, rng.random(n))
        noise2 = np.where(is_river, MASKED, rng.random(n))
        real_cause = is_river.astype(np.float64)
        control = rng.random(n)
        target = 20.0 * real_cause + rng.normal(scale=0.5, size=n)

        X = np.stack([noise1, noise2, real_cause, control], axis=1).astype(np.float32)
        y = np.stack([target, target], axis=1).astype(np.float32)

        torch.manual_seed(0)
        net = AdvantageNet(input_dim=4, hidden_sizes=(32, 32), output_dim=2, dropout=0.0)
        optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        net.train()
        train_rng = np.random.default_rng(0)
        for _ in range(1200):
            idx = train_rng.integers(0, n, size=256)
            pred = net(Xt[idx])
            loss = ((pred - yt[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        result = dict(mean_shap_contributions_for_samples(
            net, X, X, keys, np.random.default_rng(1), sample_size=150, background_size=20, nsamples=15,
        ))

        control_floor = result["hole_suited"]
        # Generous margin above the honest control's own noise-floor score
        # -- the old (unfixed) behavior scored these an order of magnitude
        # higher than a feature that's never masked and never matters.
        assert result["draw_norm"] <= control_floor * 3 + 0.05
        assert result["hole_hand_grid_x_norm"] <= control_floor * 3 + 0.05


class TestMeanShapContributionsForSamplesExplainGroupLabels:
    """Coverage for "a sub-strategy's own SHAP view should assume its
    parent's own Split By grouping is already priced in" (see
    cfr_explorer._group_labels_for_rows/_render_substrategy): a feature the
    parent's grouping fully determines (e.g. Hole Suited, fully determined
    by Exact Hole Hand) should score exactly 0 for any sub-strategy
    underneath it, not share credit with the grouping feature itself."""

    def test_feature_fully_determined_by_the_group_scores_exactly_zero(self):
        rng = np.random.default_rng(0)
        n = 2000
        # `group` stands in for a parent's own Split By feature (e.g.
        # Exact Hole Hand); `derived` stands in for a feature it fully
        # determines (e.g. Hole Suited) -- literally the same reading.
        # `control` is an honest, uncorrelated feature.
        group = (rng.random(n) < 0.5).astype(np.float64)
        derived = group.copy()
        control = rng.random(n)
        target = 15.0 * group + rng.normal(scale=0.5, size=n)

        X = np.stack([group, derived, control], axis=1).astype(np.float32)
        y = np.stack([target, target], axis=1).astype(np.float32)
        keys = ("street_norm", "hole_suited", "spr_norm")  # real, non-maskable keys; standing in for group/derived/control

        torch.manual_seed(0)
        net = AdvantageNet(input_dim=3, hidden_sizes=(32, 32), output_dim=2, dropout=0.0)
        optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        net.train()
        train_rng = np.random.default_rng(0)
        for _ in range(1200):
            idx = train_rng.integers(0, n, size=256)
            pred = net(Xt[idx])
            loss = ((pred - yt[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()

        explain_group_labels = group.astype(str)

        ungrouped = dict(mean_shap_contributions_for_samples(
            net, X, X, keys, np.random.default_rng(0), sample_size=200, background_size=20, nsamples=10,
        ))
        grouped = dict(mean_shap_contributions_for_samples(
            net, X, X, keys, np.random.default_rng(0), sample_size=200, background_size=20, nsamples=10,
            explain_group_labels=explain_group_labels,
        ))

        # Without grouping, `hole_suited` (a perfect duplicate of the
        # genuinely important `street_norm`) shares in that importance --
        # the entangled-feature problem this feature is meant to fix.
        assert ungrouped["hole_suited"] > 1.0
        # Grouped by the parent's own feature, both it and the feature it
        # fully determines become an exactly-constant reading within any
        # one (masking pattern, group) stratum -- background and explain
        # alike -- so they score exactly 0.
        assert grouped["street_norm"] == 0.0
        assert grouped["hole_suited"] == 0.0
        # The honest, uncorrelated control feature stays far below the
        # entangled features' *ungrouped* score -- not pinned to an exact
        # value, since restricting background to a same-group subset (a
        # smaller, less varied interpolation pool) shifts a GradientExplainer
        # reading by some amount for any feature, grouped or not.
        assert grouped["spr_norm"] < ungrouped["hole_suited"]

    def test_group_labels_none_takes_the_same_code_path_as_omitting_it(self):
        # Both should skip the stratify-by-group branch entirely (see
        # `if explain_group_labels is None`) -- not compared for exact
        # numeric equality, since shap.GradientExplainer's own internal
        # sampling advances global torch RNG state between calls, so two
        # calls in the same process never reproduce bit-for-bit regardless
        # of this function's own logic.
        rng = np.random.default_rng(0)
        net = AdvantageNet(input_dim=4, hidden_sizes=(8,))
        features = _filled_reservoir(capacity=30, feature_dim=4, rng=rng).features
        keys = ("hand_category_norm", "street_norm", "spr_norm", "raises_preflop_norm")

        with_explicit_none = dict(mean_shap_contributions_for_samples(
            net, features, features, keys, np.random.default_rng(0),
            sample_size=10, background_size=5, nsamples=5, explain_group_labels=None,
        ))
        without_the_argument = dict(mean_shap_contributions_for_samples(
            net, features, features, keys, np.random.default_rng(0),
            sample_size=10, background_size=5, nsamples=5,
        ))
        assert set(with_explicit_none) == set(without_the_argument)
        for key in with_explicit_none:
            assert with_explicit_none[key] == pytest.approx(without_the_argument[key], abs=0.05)


class TestPairwiseInteractionStrength:
    _KEYS = ("hand_category_norm", "street_norm", "spr_norm", "raises_preflop_norm")

    def _train_net(self, target_fn, n=2000, hidden=(24, 24), steps=800, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.random((n, 4)).astype(np.float32)
        y = target_fn(X).astype(np.float32)
        y = np.stack([y, y], axis=1)  # 2 output actions, both driven by the same target

        torch.manual_seed(seed)
        net = AdvantageNet(input_dim=4, hidden_sizes=hidden, output_dim=2, dropout=0.0)
        optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        net.train()
        train_rng = np.random.default_rng(seed)
        for _ in range(steps):
            idx = train_rng.integers(0, n, size=256)
            pred = net(Xt[idx])
            loss = ((pred - yt[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        return net, X

    def test_zero_for_an_additively_separable_target(self):
        # No x0*x1 cross term at all -- Delta_01 should cancel to ~0 (see
        # pairwise_interaction_strength's own docstring for why an
        # additively separable target does that exactly, in the limit of a
        # perfect fit).
        net, X = self._train_net(lambda X: 10.0 * X[:, 0] + 5.0 * X[:, 1] + X[:, 2])
        interaction, _weight = pairwise_interaction_strength(
            net, X, X, self._KEYS, np.random.default_rng(1), sample_size=60, background_size=15,
        )
        assert interaction[0, 1] == pytest.approx(0.0, abs=0.1)

    def test_higher_for_a_genuinely_interacting_pair_than_every_other_pair(self):
        net, X = self._train_net(lambda X: 10.0 * X[:, 0] * X[:, 1])  # pure interaction, no main effects at all
        interaction, _weight = pairwise_interaction_strength(
            net, X, X, self._KEYS, np.random.default_rng(1), sample_size=60, background_size=15,
        )
        other_pairs = [
            interaction[i, j] for i in range(4) for j in range(4) if i < j and {i, j} != {0, 1}
        ]
        assert interaction[0, 1] > max(other_pairs) * 2

    def test_matrix_is_symmetric_with_zero_diagonal(self):
        net = AdvantageNet(input_dim=4, hidden_sizes=(8,))
        features = _filled_reservoir(capacity=30, feature_dim=4, rng=np.random.default_rng(0)).features
        interaction, _weight = pairwise_interaction_strength(
            net, features, features, self._KEYS, np.random.default_rng(1), sample_size=10, background_size=5,
        )
        assert np.allclose(interaction, interaction.T)
        assert np.allclose(np.diag(interaction), 0.0)


class TestInteractionStrengthForFeature:
    _KEYS = ("hand_category_norm", "street_norm", "spr_norm", "raises_preflop_norm")

    def test_empty_pool_returns_one_entry_per_other_feature_at_zero(self):
        net = AdvantageNet(input_dim=4, hidden_sizes=(8,))
        empty = np.zeros((0, 4), dtype=np.float32)
        result = interaction_strength_for_feature(net, empty, empty, self._KEYS, "street_norm", np.random.default_rng(0))
        assert sorted(k for k, _ in result) == sorted(k for k in self._KEYS if k != "street_norm")
        assert all(v == 0.0 for _, v in result)

    def test_agrees_with_pairwise_interaction_strengths_own_row(self):
        rng = np.random.default_rng(0)
        net = AdvantageNet(input_dim=4, hidden_sizes=(16, 16))
        features = _filled_reservoir(capacity=200, feature_dim=4, rng=rng).features
        focus_key, focus_idx = self._KEYS[1], 1

        full_matrix, _weight = pairwise_interaction_strength(
            net, features, features, self._KEYS, np.random.default_rng(1), sample_size=40, background_size=10,
        )
        one_row = dict(interaction_strength_for_feature(
            net, features, features, self._KEYS, focus_key, np.random.default_rng(1),
            sample_size=40, background_size=10,
        ))
        for i, key in enumerate(self._KEYS):
            if key == focus_key:
                continue
            assert one_row[key] == pytest.approx(full_matrix[focus_idx, i], abs=1e-9)

    def test_sorted_descending(self):
        net = AdvantageNet(input_dim=4, hidden_sizes=(8,))
        features = _filled_reservoir(capacity=30, feature_dim=4, rng=np.random.default_rng(0)).features
        result = interaction_strength_for_feature(
            net, features, features, self._KEYS, "street_norm", np.random.default_rng(1),
        )
        values = [v for _, v in result]
        assert values == sorted(values, reverse=True)
