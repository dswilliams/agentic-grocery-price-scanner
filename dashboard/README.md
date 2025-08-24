# 🛒 Agentic Grocery Scanner Dashboard

A comprehensive Streamlit dashboard showcasing the complete multi-agent system with real-time monitoring of the 11-stage pipeline orchestrating 35+ nodes.

## 🚀 Features

### System Overview
- **🎯 Master Workflow**: 11-stage LangGraph pipeline visualization
- **🤖 3 Specialized Agents**: IntelligentScraperAgent, MatcherAgent, OptimizerAgent
- **📊 60+ State Fields**: Comprehensive tracking and monitoring
- **⚡ <90s Execution**: Production-ready performance for 50+ ingredient workflows

### Live Execution Monitor
- **Real-time Pipeline Visualization**: Monitor 11 stages in real-time
- **Agent Status Tracking**: Live monitoring of all 3 agents
- **Progress Indicators**: Stage-by-stage completion tracking
- **Resource Monitoring**: Memory usage and performance metrics
- **Error Recovery Visualization**: 3-tier recovery system display

### Results Dashboard
- **Multi-store Shopping Lists**: Optimized recommendations across stores
- **Strategy Comparison**: Compare 6 optimization strategies
- **Savings Analysis**: Cost breakdown and potential savings
- **Product Match Quality**: 4-tier confidence scoring system
- **Export Functionality**: PDF, CSV, and clipboard export options

### Performance Analytics
- **Execution Metrics**: Success rates, timing, and resource usage
- **Agent Performance**: Individual agent success rates and response times
- **Trend Analysis**: Historical performance over time
- **Memory Usage Breakdown**: Component-wise resource utilization

### Demo Mode
- **5 Pre-configured Scenarios**: Quick Shopping, Family Dinner, Meal Prep, Party Planning, Multi-Recipe
- **Instant Execution**: One-click demo launches
- **Performance Expectations**: Clear expectations for each scenario
- **Scenario Comparison**: Side-by-side comparison of complexity levels

## 🎯 Dashboard Pages

1. **🏠 System Overview**: Architecture and capabilities showcase
2. **📝 Recipe Input**: Multi-format ingredient and recipe entry
3. **⚡ Live Execution**: Real-time pipeline monitoring
4. **📊 Results Dashboard**: Detailed shopping recommendations
5. **📈 Performance Analytics**: System performance and metrics
6. **🎬 Demo Mode**: Pre-configured demonstration scenarios

## 🚀 Quick Start

### Launch Dashboard
```bash
# From project root
python3 launch_dashboard.py

# Or directly with Streamlit
streamlit run dashboard/streamlit_app.py
```

### Access Dashboard
- **URL**: http://localhost:8501
- **Browser**: Automatically opens in default browser
- **Mobile**: Responsive design works on mobile devices

## 📊 Demo Scenarios

### 1. Quick Shopping (3 ingredients)
- **Items**: milk, bread, eggs
- **Budget**: $25
- **Strategy**: convenience
- **Expected Time**: ~25s
- **Memory Usage**: ~250MB

### 2. Family Dinner (6 ingredients)
- **Items**: chicken breast, rice, broccoli, olive oil, garlic, onions
- **Budget**: $50
- **Strategy**: balanced
- **Expected Time**: ~45s
- **Memory Usage**: ~350MB

### 3. Meal Prep (11 ingredients)
- **Items**: ground turkey, quinoa, sweet potatoes, spinach, bell peppers, black beans, greek yogurt, bananas, oatmeal, almonds, salmon
- **Budget**: $100
- **Strategy**: quality_first
- **Expected Time**: ~65s
- **Memory Usage**: ~420MB

### 4. Party Planning (13 ingredients)
- **Items**: ground beef, tortilla chips, cheese, tomatoes, avocados, lettuce, sour cream, salsa, beer, soda, ice cream, cookies, bread
- **Budget**: $150
- **Strategy**: cost_only
- **Expected Time**: ~85s
- **Memory Usage**: ~480MB

### 5. Multi-Recipe Complex (15+ ingredients)
- **Recipes**: Chicken Stir Fry + Breakfast Smoothie Bowl
- **Budget**: $200
- **Strategy**: adaptive
- **Expected Time**: ~90s
- **Memory Usage**: ~500MB

## 🔧 Technical Details

### Dependencies
- **Streamlit**: 1.28.0+ for web interface
- **Plotly**: 5.17.0+ for interactive visualizations
- **Pandas**: 2.1.0+ for data handling
- **Multi-agent System**: Complete integration with GroceryWorkflow

### Architecture
- **Real-time Executor**: Manages workflow execution with progress callbacks
- **Dashboard Config**: Centralized configuration and styling
- **Multi-page Navigation**: Streamlit navigation for organized interface
- **State Management**: Session state for execution tracking
- **Progress Callbacks**: Real-time updates during execution

### Performance
- **Memory Efficient**: <500MB for complex workflows
- **Fast Execution**: <90s for 50+ ingredient scenarios
- **Responsive UI**: Real-time updates without blocking
- **Error Handling**: Graceful degradation and recovery

## 📈 Monitoring Features

### Real-time Metrics
- **Execution Time**: Live timing for each stage
- **Memory Usage**: Real-time memory consumption tracking
- **Success Rates**: Per-agent success rate monitoring
- **Product Collection**: Live count of products found
- **Match Quality**: Real-time confidence scoring

### Pipeline Visualization
- **Stage Progress**: 11-stage pipeline with live updates
- **Agent Status**: Color-coded status indicators
- **Error Recovery**: Visual representation of recovery processes
- **Performance Graphs**: Interactive charts and metrics

### Export Options
- **Shopping Lists**: PDF and CSV export
- **Performance Reports**: Detailed execution analytics
- **Configuration**: Save and load custom configurations
- **History**: Execution history tracking

## 🎮 Interactive Elements

### Input Methods
- **Individual Ingredients**: Dynamic form with quantity/units
- **Recipe Format**: Structured recipe entry with parsing
- **Bulk Text**: Comma/newline separated ingredient lists
- **Demo Scenarios**: One-click pre-configured scenarios

### Customization
- **Budget Constraints**: Flexible budget limits
- **Store Selection**: Multi-store selection with preferences
- **Strategy Selection**: 6 optimization strategies
- **Quality Thresholds**: Configurable confidence thresholds

### Real-time Controls
- **Start/Stop Execution**: Real-time execution control
- **Strategy Switching**: Dynamic strategy adjustment
- **Store Filtering**: Live store selection changes
- **Export Options**: Multiple export formats

## 🛡️ Error Handling

### Graceful Degradation
- **Agent Failures**: Partial results with clear error messages
- **Network Issues**: Offline mode with cached data
- **Memory Limits**: Automatic optimization for resource constraints
- **Import Failures**: Fallback modes for missing dependencies

### User Feedback
- **Clear Error Messages**: Detailed error descriptions with solutions
- **Recovery Suggestions**: Actionable steps for error resolution
- **Progress Indicators**: Clear feedback during all operations
- **Status Updates**: Real-time status communication

## 📋 Usage Examples

### Basic Workflow
1. **Navigate to Recipe Input**: Enter ingredients or recipes
2. **Configure Preferences**: Set budget, stores, and strategy
3. **Execute Workflow**: Click "Execute Workflow" button
4. **Monitor Progress**: Watch real-time execution in Live Execution page
5. **View Results**: Examine optimized shopping lists in Results Dashboard
6. **Analyze Performance**: Check execution metrics in Performance Analytics

### Demo Mode Usage
1. **Navigate to Demo Mode**: Select from 5 pre-configured scenarios
2. **Choose Scenario**: Pick complexity level and use case
3. **Run Demo**: One-click execution with expected performance
4. **Monitor Execution**: Real-time monitoring of demo workflow
5. **Compare Results**: Analyze results across different scenarios

### Performance Monitoring
1. **System Overview**: Check agent health and system status
2. **Live Execution**: Monitor resource usage during execution
3. **Performance Analytics**: Review historical performance trends
4. **Export Reports**: Generate performance reports for analysis

This dashboard provides a comprehensive view of the sophisticated multi-agent orchestration system, showcasing the power and efficiency of the 11-stage pipeline with real-time monitoring and professional visualization.