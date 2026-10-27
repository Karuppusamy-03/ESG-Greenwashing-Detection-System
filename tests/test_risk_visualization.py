import unittest

import app


class RiskVisualizationTests(unittest.TestCase):
    def test_distribution_chart_counts_only_predicted_categories(self):
        figure = app.build_risk_distribution_figure([
            {"demonstration_risk_category": "LOW"},
            {"demonstration_risk_category": "LOW"},
            {"demonstration_risk_category": "HIGH"},
            {"demonstration_risk_category": "NOT_PREDICTED"},
            {},
            None,
        ])

        self.assertIsNotNone(figure)
        self.assertEqual(
            [trace.name for trace in figure.data],
            ["LOW", "MEDIUM", "HIGH"],
        )
        self.assertEqual(
            [int(trace.y[0]) for trace in figure.data],
            [2, 0, 1],
        )
        self.assertEqual(
            figure.layout.yaxis.title.text,
            "Claims",
        )

    def test_distribution_chart_has_empty_state_for_no_predictions(self):
        self.assertIsNone(app.build_risk_distribution_figure([]))
        self.assertIsNone(app.build_risk_distribution_figure([
            {"demonstration_risk_category": "NOT_PREDICTED"},
            {"demonstration_risk_category": None},
        ]))


if __name__ == "__main__":
    unittest.main()
