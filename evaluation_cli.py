"""
Evaluation Framework CLI

Command-line interface for the comprehensive evaluation framework including
quality monitoring, regression testing, business metrics, and reporting.
"""

import asyncio
import json
import click
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

from evaluation.quality_monitor import QualityMonitor, ContinuousQualityMonitor
from evaluation.regression_tester import RegressionTester, ContinuousRegressionTester
from evaluation.ml_model_evaluator import MLModelEvaluator
from evaluation.business_metrics_validator import BusinessMetricsValidator
from evaluation.continuous_improvement import ContinuousImprovementPipeline
from evaluation.reporting_system import ComprehensiveReportingSystem, ReportType, ReportPriority
from evaluation.golden_dataset import GoldenDatasetManager


@click.group()
def cli():
    """
    🔍 Evaluation Framework CLI
    
    Comprehensive quality monitoring and evaluation system for the 
    Agentic Grocery Price Scanner.
    """
    pass


@cli.group()
def quality():
    """Quality monitoring and assessment commands."""
    pass


@quality.command('assess')
@click.option('--save', '-s', is_flag=True, help='Save results to file')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
def quality_assess(save, verbose):
    """Run comprehensive quality assessment."""
    click.echo("🔍 Running comprehensive quality assessment...")
    
    async def run_assessment():
        monitor = QualityMonitor()
        report = await monitor.run_comprehensive_evaluation()
        
        # Display results
        click.echo(f"\n📊 Quality Assessment Results:")
        click.echo(f"Overall Score: {report.overall_score:.1f}/100")
        
        if verbose:
            click.echo(f"Component Scores: {report.component_scores}")
            click.echo(f"Alerts: {len(report.alerts)}")
            click.echo(f"Recommendations: {len(report.recommendations)}")
            
            if report.alerts:
                click.echo(f"\n⚠️ Active Alerts:")
                for alert in report.alerts:
                    click.echo(f"  {alert['severity'].upper()}: {alert['message']}")
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation/results/quality_assessment_{timestamp}.json"
            
            # Save report
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({
                    'overall_score': report.overall_score,
                    'component_scores': report.component_scores,
                    'alerts': report.alerts,
                    'recommendations': report.recommendations,
                    'timestamp': report.timestamp.isoformat()
                }, f, indent=2, default=str)
            
            click.echo(f"✅ Results saved to {output_file}")
    
    asyncio.run(run_assessment())


@quality.command('monitor')
@click.option('--interval', '-i', default=6, help='Monitoring interval in hours')
@click.option('--duration', '-d', default=24, help='Total monitoring duration in hours')
def quality_monitor(interval, duration):
    """Start continuous quality monitoring."""
    click.echo(f"🚀 Starting continuous quality monitoring...")
    click.echo(f"Interval: {interval} hours, Duration: {duration} hours")
    
    async def run_monitoring():
        monitor = ContinuousQualityMonitor(monitor_interval_hours=interval)
        
        # Run for specified duration
        end_time = datetime.now() + timedelta(hours=duration)
        
        try:
            while datetime.now() < end_time:
                report = await monitor.quality_monitor.run_comprehensive_evaluation()
                
                click.echo(f"\n📊 Quality Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                click.echo(f"Overall Score: {report.overall_score:.1f}/100")
                
                if report.alerts:
                    critical_alerts = [a for a in report.alerts if a['severity'] == 'critical']
                    if critical_alerts:
                        click.echo(f"🚨 {len(critical_alerts)} critical alerts!")
                
                await asyncio.sleep(interval * 3600)
                
        except KeyboardInterrupt:
            click.echo("\n⏹️ Monitoring stopped by user")
    
    asyncio.run(run_monitoring())


@cli.group()
def regression():
    """Regression testing commands."""
    pass


@regression.command('test')
@click.option('--save', '-s', is_flag=True, help='Save results to file')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
def regression_test(save, verbose):
    """Run regression test suite."""
    click.echo("🧪 Running comprehensive regression tests...")
    
    async def run_tests():
        tester = RegressionTester()
        suite = await tester.run_full_regression_suite()
        
        # Display results
        click.echo(f"\n🧪 Regression Test Results:")
        click.echo(f"Health Score: {suite.overall_health_score:.1f}/100")
        click.echo(f"Tests Passed: {suite.passed_tests}/{suite.total_tests}")
        click.echo(f"Regressions: {suite.regressions_detected}")
        
        if verbose:
            click.echo(f"Execution Time: {suite.execution_time:.2f}s")
            
            if suite.critical_regressions > 0:
                click.echo(f"🚨 Critical Regressions: {suite.critical_regressions}")
            if suite.major_regressions > 0:
                click.echo(f"⚠️ Major Regressions: {suite.major_regressions}")
            if suite.minor_regressions > 0:
                click.echo(f"ℹ️ Minor Regressions: {suite.minor_regressions}")
        
        if save:
            tester.save_results(suite)
            click.echo("✅ Results saved to file")
    
    asyncio.run(run_tests())


@regression.command('monitor')
@click.option('--interval', '-i', default=12, help='Testing interval in hours')
def regression_monitor(interval):
    """Start continuous regression testing."""
    click.echo(f"🚀 Starting continuous regression testing...")
    click.echo(f"Interval: {interval} hours")
    
    async def run_monitoring():
        tester = ContinuousRegressionTester(test_interval_hours=interval)
        
        try:
            await tester.start_continuous_testing()
        except KeyboardInterrupt:
            click.echo("\n⏹️ Testing stopped by user")
    
    asyncio.run(run_monitoring())


@cli.group()
def ml():
    """ML model evaluation commands."""
    pass


@ml.command('evaluate')
@click.option('--models', '-m', default='all', help='Models to evaluate (llm, embedding, all)')
@click.option('--save', '-s', is_flag=True, help='Save results to file')
def ml_evaluate(models, save):
    """Evaluate ML model performance."""
    click.echo("🤖 Evaluating ML models...")
    
    async def run_evaluation():
        evaluator = MLModelEvaluator()
        
        if models == 'all':
            results = await evaluator.run_comprehensive_evaluation()
        else:
            # Individual model evaluation would go here
            results = await evaluator.run_comprehensive_evaluation()
        
        # Display results
        click.echo(f"\n🤖 ML Model Evaluation Results:")
        
        for model_name, result in results.items():
            click.echo(f"\n{model_name.upper()} Model:")
            click.echo(f"  Overall Score: {result.overall_score:.1f}/100")
            click.echo(f"  Drift Detected: {'Yes' if result.drift_detected else 'No'}")
            click.echo(f"  Retraining Recommended: {'Yes' if result.retraining_recommended else 'No'}")
            
            if result.retraining_recommended:
                click.echo(f"  Urgency: {result.retraining_urgency}")
        
        if save:
            evaluator.save_results(results)
            click.echo("✅ Results saved to file")
    
    asyncio.run(run_evaluation())


@cli.group()
def business():
    """Business metrics validation commands."""
    pass


@business.command('validate')
@click.option('--period', '-p', default=30, help='Analysis period in days')
@click.option('--save', '-s', is_flag=True, help='Save results to file')
def business_validate(period, save):
    """Validate business metrics and impact."""
    click.echo(f"💼 Validating business metrics for {period}-day period...")
    
    def run_validation():
        validator = BusinessMetricsValidator()
        report = validator.validate_business_metrics(period_days=period)
        
        # Display results
        click.echo(f"\n💼 Business Metrics Validation:")
        click.echo(f"Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}")
        click.echo(f"Total Sessions: {report.total_sessions}")
        click.echo(f"Average Satisfaction: {report.avg_satisfaction_score:.2f}/5.0")
        click.echo(f"Total Savings: ${report.total_actual_savings:.2f}")
        click.echo(f"Savings Accuracy: {report.savings_accuracy_rate:.1%}")
        click.echo(f"ROI: {report.roi_percentage:.1f}%")
        
        if report.risk_factors:
            click.echo(f"\n⚠️ Risk Factors ({len(report.risk_factors)}):")
            for risk in report.risk_factors:
                click.echo(f"  • {risk}")
        
        if save:
            validator.save_validation_report(report)
            click.echo("✅ Results saved to file")
    
    run_validation()


@cli.group()
def improvement():
    """Continuous improvement commands."""
    pass


@improvement.command('analyze')
@click.option('--save', '-s', is_flag=True, help='Save improvement plan')
@click.option('--implement', is_flag=True, help='Attempt automated implementations')
def improvement_analyze(save, implement):
    """Analyze system and generate improvement recommendations."""
    click.echo("🔄 Analyzing system for improvement opportunities...")
    
    async def run_analysis():
        pipeline = ContinuousImprovementPipeline()
        plan = await pipeline.run_improvement_cycle()
        
        # Display results
        summary = pipeline.generate_improvement_summary(plan)
        
        click.echo(f"\n🔄 Improvement Analysis Results:")
        click.echo(f"Total Recommendations: {summary['total_recommendations']}")
        click.echo(f"Quick Wins: {summary['quick_wins']}")
        click.echo(f"Priority Items: {summary['priority_items']}")
        click.echo(f"Estimated Effort: {summary['estimated_total_effort']:.1f} hours")
        
        click.echo(f"\n⭐ Top Recommendations:")
        for i, rec in enumerate(summary['top_recommendations'], 1):
            click.echo(f"  {i}. {rec['title']} ({rec['priority']} priority)")
            click.echo(f"     Impact: {rec['estimated_impact']:.0f}%, Complexity: {rec['complexity']}/5")
        
        if save:
            pipeline.save_improvement_plan(plan)
            click.echo("✅ Improvement plan saved")
    
    asyncio.run(run_analysis())


@improvement.command('monitor')
@click.option('--interval', '-i', default=24, help='Analysis interval in hours')
def improvement_monitor(interval):
    """Start continuous improvement monitoring."""
    click.echo(f"🚀 Starting continuous improvement monitoring...")
    click.echo(f"Interval: {interval} hours")
    
    async def run_monitoring():
        pipeline = ContinuousImprovementPipeline()
        
        try:
            await pipeline.start_continuous_monitoring(interval_hours=interval)
        except KeyboardInterrupt:
            click.echo("\n⏹️ Monitoring stopped by user")
    
    asyncio.run(run_monitoring())


@cli.group()
def report():
    """Report generation and distribution commands."""
    pass


@report.command('generate')
@click.option('--type', '-t', 
              type=click.Choice(['daily', 'weekly', 'monthly', 'quarterly']),
              required=True,
              help='Type of report to generate')
@click.option('--priority', '-p',
              type=click.Choice(['low', 'normal', 'high', 'urgent']),
              default='normal',
              help='Report priority level')
@click.option('--notify/--no-notify', default=True, help='Send notifications to stakeholders')
def report_generate(type, priority, notify):
    """Generate and distribute reports."""
    
    type_mapping = {
        'daily': ReportType.DAILY_SUMMARY,
        'weekly': ReportType.WEEKLY_QUALITY,
        'monthly': ReportType.MONTHLY_BUSINESS,
        'quarterly': ReportType.QUARTERLY_EXECUTIVE
    }
    
    priority_mapping = {
        'low': ReportPriority.LOW,
        'normal': ReportPriority.NORMAL,
        'high': ReportPriority.HIGH,
        'urgent': ReportPriority.URGENT
    }
    
    click.echo(f"📊 Generating {type} report with {priority} priority...")
    
    async def run_report_generation():
        reporting_system = ComprehensiveReportingSystem()
        
        if notify:
            report = await reporting_system.generate_and_distribute_report(
                type_mapping[type],
                priority_mapping[priority]
            )
            click.echo(f"✅ Report generated and distributed to {len(report.stakeholders_notified)} stakeholders")
        else:
            report = await reporting_system.report_generator.generate_report(type_mapping[type])
            click.echo(f"✅ Report generated (no notifications sent)")
        
        click.echo(f"Report ID: {report.report_id}")
        click.echo(f"Generation Time: {report.generation_time:.2f}s")
        
        if report.html_file:
            click.echo(f"HTML Report: {report.html_file}")
        if report.json_file:
            click.echo(f"JSON Report: {report.json_file}")
    
    asyncio.run(run_report_generation())


@cli.group()
def dataset():
    """Golden dataset management commands."""
    pass


@dataset.command('stats')
def dataset_stats():
    """Show golden dataset statistics."""
    manager = GoldenDatasetManager()
    stats = manager.get_dataset_stats()
    
    click.echo("📊 Golden Dataset Statistics:")
    click.echo(f"Total Matches: {stats['total_matches']}")
    click.echo(f"Categories: {stats['categories']}")
    click.echo(f"Stores: {stats['stores']}")
    click.echo(f"Edge Cases: {stats['edge_cases']} ({stats['edge_case_percentage']:.1f}%)")
    click.echo(f"Need Verification: {stats['needs_verification']}")


@dataset.command('verify')
@click.option('--category', '-c', help='Verify specific category only')
def dataset_verify(category):
    """Show matches that need verification."""
    manager = GoldenDatasetManager()
    
    if category:
        matches = manager.get_matches_by_category(category)
        needing_verification = [m for m in matches if m.needs_verification()]
    else:
        needing_verification = manager.get_matches_needing_verification()
    
    if needing_verification:
        click.echo(f"📝 {len(needing_verification)} matches need verification:")
        for match in needing_verification[:10]:  # Show first 10
            days_old = (datetime.now().date() - match.last_verified).days
            click.echo(f"  • {match.ingredient_name} → {match.product_name} (verified {days_old} days ago)")
    else:
        click.echo("✅ All matches are up to date!")


@cli.command('dashboard')
@click.option('--port', '-p', default=8501, help='Dashboard port')
def dashboard(port):
    """Launch quality monitoring dashboard."""
    click.echo(f"🎛️ Launching quality monitoring dashboard on port {port}...")
    click.echo("🌐 Dashboard will open at http://localhost:{port}")
    
    try:
        import subprocess
        subprocess.run([
            'streamlit', 'run', 
            'evaluation/quality_dashboard.py', 
            '--server.port', str(port)
        ])
    except FileNotFoundError:
        click.echo("❌ Streamlit not found. Install with: pip install streamlit")
    except Exception as e:
        click.echo(f"❌ Error launching dashboard: {e}")


@cli.command('status')
def status():
    """Show overall system evaluation status."""
    click.echo("🔍 System Evaluation Status Overview")
    click.echo("=" * 50)
    
    async def show_status():
        # Quick status check
        try:
            # Quality check
            monitor = QualityMonitor()
            quality_report = await monitor.run_comprehensive_evaluation()
            
            click.echo(f"📊 Quality Score: {quality_report.overall_score:.1f}/100")
            
            if quality_report.overall_score >= 90:
                click.echo("✅ System Health: EXCELLENT")
            elif quality_report.overall_score >= 75:
                click.echo("⚠️ System Health: GOOD (Some issues)")
            else:
                click.echo("🚨 System Health: NEEDS ATTENTION")
            
            click.echo(f"📋 Active Alerts: {len(quality_report.alerts)}")
            click.echo(f"💡 Recommendations: {len(quality_report.recommendations)}")
            
            # Business metrics
            business_validator = BusinessMetricsValidator()
            business_report = business_validator.validate_business_metrics(7)  # Last 7 days
            
            click.echo(f"💼 User Satisfaction: {business_report.avg_satisfaction_score:.1f}/5.0")
            click.echo(f"💰 ROI: {business_report.roi_percentage:.1f}%")
            
        except Exception as e:
            click.echo(f"❌ Error getting status: {e}")
    
    asyncio.run(show_status())


@cli.command('full-evaluation')
@click.option('--save-all', is_flag=True, help='Save all results to files')
@click.option('--generate-report', is_flag=True, help='Generate comprehensive report')
def full_evaluation(save_all, generate_report):
    """Run complete evaluation suite (quality, regression, ML, business)."""
    click.echo("🚀 Running complete evaluation suite...")
    click.echo("This may take several minutes...")
    
    async def run_full_evaluation():
        results = {}
        
        # Quality Assessment
        click.echo("\n1️⃣ Running quality assessment...")
        monitor = QualityMonitor()
        quality_report = await monitor.run_comprehensive_evaluation()
        results['quality'] = quality_report
        click.echo(f"   ✅ Quality Score: {quality_report.overall_score:.1f}/100")
        
        # Regression Testing
        click.echo("\n2️⃣ Running regression tests...")
        tester = RegressionTester()
        regression_suite = await tester.run_full_regression_suite()
        results['regression'] = regression_suite
        click.echo(f"   ✅ Test Pass Rate: {regression_suite.passed_tests}/{regression_suite.total_tests}")
        
        # ML Model Evaluation
        click.echo("\n3️⃣ Evaluating ML models...")
        ml_evaluator = MLModelEvaluator()
        ml_evaluation = await ml_evaluator.run_comprehensive_evaluation()
        results['ml'] = ml_evaluation
        click.echo(f"   ✅ Models Evaluated: {len(ml_evaluation)}")
        
        # Business Metrics
        click.echo("\n4️⃣ Validating business metrics...")
        business_validator = BusinessMetricsValidator()
        business_report = business_validator.validate_business_metrics()
        results['business'] = business_report
        click.echo(f"   ✅ Sessions Analyzed: {business_report.total_sessions}")
        
        # Improvement Analysis
        click.echo("\n5️⃣ Analyzing improvements...")
        improvement_pipeline = ContinuousImprovementPipeline()
        improvement_plan = await improvement_pipeline.run_improvement_cycle()
        results['improvements'] = improvement_plan
        click.echo(f"   ✅ Recommendations: {len(improvement_plan.recommendations)}")
        
        # Summary
        click.echo("\n📊 Complete Evaluation Summary:")
        click.echo(f"Overall Quality Score: {quality_report.overall_score:.1f}/100")
        click.echo(f"System Health: {'HEALTHY' if quality_report.overall_score >= 85 else 'NEEDS ATTENTION'}")
        click.echo(f"Regressions Detected: {regression_suite.regressions_detected}")
        click.echo(f"Business ROI: {business_report.roi_percentage:.1f}%")
        click.echo(f"Improvement Opportunities: {len(improvement_plan.recommendations)}")
        
        if save_all:
            click.echo("\n💾 Saving all results...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save individual results
            tester.save_results(regression_suite)
            ml_evaluator.save_results(ml_evaluation)
            business_validator.save_validation_report(business_report)
            improvement_pipeline.save_improvement_plan(improvement_plan)
            
            click.echo("✅ All results saved to evaluation/results/")
        
        if generate_report:
            click.echo("\n📄 Generating comprehensive report...")
            reporting_system = ComprehensiveReportingSystem()
            report = await reporting_system.generate_and_distribute_report(
                ReportType.WEEKLY_QUALITY,
                ReportPriority.HIGH
            )
            click.echo(f"✅ Report generated: {report.html_file}")
    
    asyncio.run(run_full_evaluation())


if __name__ == '__main__':
    cli()