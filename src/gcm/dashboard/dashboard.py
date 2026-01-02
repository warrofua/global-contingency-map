"""
Dashboard for regime visualization
Uses Plotly for interactive visualizations
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class RegimeDashboard:
    """
    Interactive dashboard for regime monitoring

    Displays:
    - Current regime state and probabilities
    - Early warning signals (AR1, variance, flickering)
    - Historical regime transitions
    - Transition probability forecasts
    """

    def __init__(self):
        """Initialize dashboard"""
        self.regime_colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
        logger.info("Initialized regime dashboard")

    def create_daily_report(
        self,
        pipeline_output: Dict,
        historical_data: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Create daily monitoring dashboard

        Args:
            pipeline_output: Output from FinancialSurfacePipeline.run_daily_update()
            historical_data: Optional historical backtest data

        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Current Regime Probabilities',
                'Transition Forecast (20 days)',
                'Early Warning: AR(1) Autocorrelation',
                'Early Warning: Variance Ratio',
                'Historical Regimes',
                'Alert Level'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter", "colspan": 2}, None]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )

        # 1. Current regime probabilities
        regime_info = pipeline_output['regime']
        if regime_info['state_probabilities'] is not None:
            probs = regime_info['state_probabilities']
            fig.add_trace(
                go.Bar(
                    x=[f"Regime {i}" for i in range(len(probs))],
                    y=probs,
                    marker_color=self.regime_colors[:len(probs)],
                    name="Current State"
                ),
                row=1, col=1
            )

        # 2. Transition forecast
        if regime_info['transition_20day'] is not None:
            trans_probs = regime_info['transition_20day']
            fig.add_trace(
                go.Bar(
                    x=[f"Regime {i}" for i in range(len(trans_probs))],
                    y=trans_probs,
                    marker_color=self.regime_colors[:len(trans_probs)],
                    name="20-Day Forecast"
                ),
                row=1, col=2
            )

        # 3-6. Historical data plots
        if historical_data is not None and len(historical_data) > 0:
            dates = historical_data.index

            # 3. AR1 over time
            if 'ar1' in historical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=historical_data['ar1'],
                        mode='lines',
                        name='AR(1)',
                        line=dict(color='#2E86AB')
                    ),
                    row=2, col=1
                )
                # Add threshold lines
                fig.add_hline(y=0.7, line_dash="dash", line_color="orange", row=2, col=1)
                fig.add_hline(y=0.8, line_dash="dash", line_color="red", row=2, col=1)

            # 4. Variance ratio
            if 'variance_ratio' in historical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=historical_data['variance_ratio'],
                        mode='lines',
                        name='Variance Ratio',
                        line=dict(color='#A23B72')
                    ),
                    row=2, col=2
                )
                fig.add_hline(y=2.0, line_dash="dash", line_color="orange", row=2, col=2)

            # 5. Historical regimes
            if 'regime' in historical_data.columns:
                # Color by regime
                colors = [self.regime_colors[int(r) % len(self.regime_colors)]
                         if not np.isnan(r) else '#CCCCCC'
                         for r in historical_data['regime']]

                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=historical_data['regime'],
                        mode='markers',
                        name='Regime',
                        marker=dict(color=colors, size=8)
                    ),
                    row=3, col=1
                )

            # 6. Alert level (on same plot as regime)
            if 'alert_level' in historical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=historical_data['alert_level'],
                        mode='lines',
                        name='Alert Level',
                        line=dict(color='#F18F01', width=2),
                        yaxis='y2'
                    ),
                    row=3, col=1
                )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Global Contingency Map - {pipeline_output['ticker']} - {pipeline_output['timestamp'].strftime('%Y-%m-%d %H:%M')}",
                x=0.5,
                xanchor='center'
            ),
            height=900,
            showlegend=True,
            template="plotly_white"
        )

        # Axis labels
        fig.update_xaxes(title_text="Regime", row=1, col=1)
        fig.update_xaxes(title_text="Regime", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_xaxes(title_text="Date", row=3, col=1)

        fig.update_yaxes(title_text="Probability", row=1, col=1)
        fig.update_yaxes(title_text="Probability", row=1, col=2)
        fig.update_yaxes(title_text="AR(1)", row=2, col=1)
        fig.update_yaxes(title_text="Variance Ratio", row=2, col=2)
        fig.update_yaxes(title_text="Regime State", row=3, col=1)

        return fig

    def create_backtest_report(
        self,
        backtest_data: pd.DataFrame,
        events: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create backtest analysis dashboard

        Args:
            backtest_data: DataFrame from pipeline.backtest()
            events: Optional dict of known events {'date': 'description'}

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                'Regime Evolution',
                'Early Warning: AR(1) Autocorrelation',
                'Early Warning: Variance',
                'Alert Level'
            ),
            vertical_spacing=0.08
        )

        dates = backtest_data.index

        # 1. Regime evolution
        if 'regime' in backtest_data.columns:
            colors = [self.regime_colors[int(r) % len(self.regime_colors)]
                     if not np.isnan(r) else '#CCCCCC'
                     for r in backtest_data['regime']]

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=backtest_data['regime'],
                    mode='markers',
                    name='Regime',
                    marker=dict(color=colors, size=6)
                ),
                row=1, col=1
            )

        # 2. AR1
        if 'ar1' in backtest_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=backtest_data['ar1'],
                    mode='lines',
                    name='AR(1)',
                    line=dict(color='#2E86AB')
                ),
                row=2, col=1
            )
            fig.add_hline(y=0.7, line_dash="dash", line_color="orange", row=2, col=1)
            fig.add_hline(y=0.8, line_dash="dash", line_color="red", row=2, col=1)

        # 3. Variance
        if 'variance' in backtest_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=backtest_data['variance'],
                    mode='lines',
                    name='Variance',
                    line=dict(color='#A23B72')
                ),
                row=3, col=1
            )

        # 4. Alert level
        if 'alert_level' in backtest_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=backtest_data['alert_level'],
                    mode='lines',
                    name='Alert Level',
                    line=dict(color='#F18F01', width=2),
                    fill='tozeroy'
                ),
                row=4, col=1
            )

        # Add event markers if provided
        if events:
            for event_date, event_desc in events.items():
                event_dt = pd.to_datetime(event_date)
                for row in range(1, 5):
                    fig.add_vline(
                        x=event_dt,
                        line_dash="dot",
                        line_color="red",
                        annotation_text=event_desc,
                        annotation_position="top",
                        row=row, col=1
                    )

        fig.update_layout(
            title="Backtest Analysis",
            height=1000,
            showlegend=True,
            template="plotly_white"
        )

        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Regime", row=1, col=1)
        fig.update_yaxes(title_text="AR(1)", row=2, col=1)
        fig.update_yaxes(title_text="Variance", row=3, col=1)
        fig.update_yaxes(title_text="Alert", row=4, col=1)

        return fig

    def save_html(self, fig: go.Figure, filename: str):
        """Save figure as HTML"""
        fig.write_html(filename)
        logger.info(f"Dashboard saved to {filename}")

    def show(self, fig: go.Figure):
        """Display figure in browser"""
        fig.show()
