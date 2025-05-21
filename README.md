# Banking Customer Retention Optimizer

![Customer Segments](reports/figures/customer_segments.png)

## Business Problem: Optimizing Retention Investment During Digital Transformation 

Atlantic Regional Bank is undergoing a major *digital transformation*, shifting resources from physical branches to digital channels. 
The bank must simultaneously reduce branch operating costs while preventing customer attrition during this transition. 
With a limited retention budget of $1.5 million, the bank needs to identify which customers are both at high risk of leaving 
AND most likely to respond positively to retention efforts. Traditional retention approaches that target all high-risk customers are not financially viable 
during this transformation period, requiring a more sophisticated, ROI-focused approach.

## Business Value & Impact

### What This Project Accomplishes
This project delivers a complete customer retention solution that enables banks to:
- Predict which customers are likely to leave with 59.4% precision
- Identify different customer segments requiring tailored retention approaches
- Allocate limited retention budgets for maximum ROI (3.2x return demonstrated)
- Implement a phased transition strategy supporting the shift to digital channels
- Continuously refine retention strategies through A/B testing

### Who Benefits
1. **Banking Executives**: Gain a strategic framework for managing digital transformation with reduced customer attrition and optimized spending.

2. **Marketing Teams**: Receive actionable customer segmentation with targeted retention strategies rather than one-size-fits-all approaches.

3. **Digital Transformation Leaders**: Obtain a data-driven roadmap for transitioning customers from branches to digital channels with minimal attrition.

4. **Customer Experience Teams**: Receive insights to develop segment-specific experiences that address the unique needs of different customer groups.

5. **Finance Departments**: Benefit from quantifiable ROI projections and budget optimization for retention initiatives.


## Key Outcomes
- Predictive model identifying customers at risk of leaving (59.4% precision, 68.1% recall)
- Customer segmentation with tailored retention strategies for digital-ready vs. branch-dependent customers
- ROI analysis showing 3.2x return on retention investment
- Phased implementation plan for retention programs

### Measurable Outcomes
The implementation of this solution provides:
- **Financial Impact**: $189,390 net benefit from retention investments
- **Operational Efficiency**: 59.4% targeting precision means reduced wasted resources
- **Customer Retention**: 68.1% of at-risk customers identified for proactive intervention
- **Digital Adoption**: Strategies for accelerating digital engagement across all customer segments
- **Risk Mitigation**: Early identification of high-value customers at risk during transformation

## Approach
This project followed a systematic approach:

1. **Data Analysis & Feature Engineering**: Understanding churn patterns and creating banking-specific predictive features
2. **Model Development & Selection**: Testing and optimizing machine learning models with focus on business metrics
3. **Business Application**: Developing targeted retention strategies with ROI assessment
4. **Implementation Framework**: Creating a practical roadmap with A/B testing approach

## Project Structure:

```
banking-retention-optimizer/
├── data/
│   └── Churn_Modelling.csv
├── notebooks/
│   ├── 1_data-analysis.ipynb              # Data exploration & visualization
│   ├── 2_model_development.ipynb          # Model selection and optimization
│   └── 3_business_application.ipynb       # Segmentation, strategies & ROI
├── reports/
│   ├── figures/                           # Visualizations
│   └── model_comparison.csv               # Model performance metrics
├── README.md                              # Project documentation
├── .gitignore
└── requirements.txt                       # Dependencies
```

### Notebook 1: Data Analysis
This notebook explores the customer dataset to understand key factors influencing churn during digital transformation:

- Identified significant age difference between churning (older) and retained customers
- Discovered inactive members are 3x more likely to leave
- Found geography plays a meaningful role in retention
- Revealed that customers with multiple products have lower churn rates
- Analyzed balance patterns showing zero-balance accounts have higher churn risk

### Notebook 2: Model Development
This notebook focuses on developing predictive models for customer churn:

- Created custom features for digital transformation context (Digital_Readiness, Retention_Score)
- Implemented feature engineering pipeline with appropriate scaling and encoding
- Evaluated multiple algorithms (Random Forest, AdaBoost, XGBoost)
- Conducted hyperparameter tuning with F1 score optimization
- Performed detailed threshold analysis to balance precision and recall for budget constraints
- Selected Random Forest (Optimized) as best model with F1 score of 0.635

### Notebook 3: Business Application
This notebook applies the model to create actionable business strategies:

- Developed customer segmentation combining churn risk and digital readiness
- Created tailored retention strategies for each segment
- Calculated ROI for retention investments by segment
- Designed a three-phase implementation roadmap
- Established an A/B testing framework for ongoing optimization

## Model Performance

The project evaluated multiple algorithms, with the following performance on the test set:

| Algorithm | F1 Score | Precision | Recall | AUC |
|-----------|----------|-----------|--------|-----|
| Random Forest (Optimized) | 0.635 | 0.594 | 0.681 | 0.861 |
| XGBoost (Optimized) | 0.618 | 0.556 | 0.695 | 0.855 |
| XGBoost | 0.573 | 0.687 | 0.491 | 0.839 |
| AdaBoost | 0.571 | 0.692 | 0.487 | 0.849 |
| AdaBoost (Optimized) | 0.568 | 0.764 | 0.452 | 0.848 |
| Random Forest | 0.554 | 0.763 | 0.435 | 0.846 |

Random Forest (Optimized) provided the best balance of precision and recall, which is critical for our limited budget scenario. The model achieves 59.4% precision (efficiency of retention spending) while capturing 68.1% of customers who would churn (reach of retention efforts).

## Key Insights

![Age Distribution](reports/figures/boxplot_Age.png)

1. **Age is a critical factor**: Customers 40-60 years old show significantly higher churn probability
2. **Digital readiness varies**: Customers with low digital readiness need specialized transition support
3. **Value-based segmentation is essential**: High-value customers require different retention approaches
4. **Product holding matters**: Multiple products significantly reduce churn risk
5. **Targeted strategies outperform**: ROI varies significantly across segments

## Customer Segments & Strategies

![ROI by Segment](reports/figures/segment_roi.png)

| Segment | Characteristics | Strategy | Expected Results |
|---------|----------------|----------|------------------|
| High-Value Digital-Ready At-Risk | Balance >$100k, High churn probability, High digital readiness | Premium Digital VIP Program | 40% churn reduction |
| High-Value Branch-Dependent At-Risk | Balance >$100k, High churn probability, Low digital readiness | Executive Transition Program | 35% churn reduction |
| Digital-Ready At-Risk | High churn probability, High digital readiness | Digital-Focused Retention Program | 30% churn reduction |
| Branch-Dependent At-Risk | High churn probability, Low digital readiness | Guided Transition Program | 25% churn reduction |
| Digital-Ready Watch | Medium churn probability, High digital readiness | Digital Engagement Program | 20% churn reduction |
| Branch-Dependent Watch | Medium churn probability, Low digital readiness | Relationship Review Program | 15% churn reduction |
| Stable Customers | Low churn probability | Value Growth Program | 10% churn reduction |

## Financial Impact

The retention program shows strong ROI potential:

- **Total Program Cost**: $86,450
- **Value Protected**: $275,840
- **Net Benefit**: $189,390
- **Return on Investment**: 3.2x

## Implementation Plan

![Implementation Phases](reports/figures/implementation_phases.png)

The retention strategy is designed for phased implementation:

1. **Phase 1: High-Risk Intervention** (Weeks 1-2)
   - Immediate focus on high-value at-risk customers
   - Personal outreach and premium offers

2. **Phase 2: Proactive Engagement** (Weeks 3-8)
   - Medium-risk customer engagement
   - Relationship reviews and cross-selling

3. **Phase 3: Value Growth** (Month 3+)
   - Long-term loyalty development
   - Referral programs and satisfaction monitoring

## A/B Testing Framework

To continuously optimize retention efforts, the project includes an A/B testing framework:

| Segment | Test Variants | Primary Metric |
|---------|--------------|---------------|
| Digital-Ready At-Risk | Standard email vs. Personalized tutorial vs. Incentive program | Digital engagement rate |
| Branch-Dependent At-Risk | Branch notification vs. Guided transition vs. Hybrid model | Digital adoption rate |

## Technical Highlights

- **Custom Feature Engineering**: Created domain-specific features for digital transformation context
- **ROI-Focused Model Selection**: Optimized for business impact rather than just predictive accuracy
- **Threshold Analysis**: Determined optimal probability threshold to maximize retention ROI
- **Segment-Specific Strategies**: Developed tailored approaches based on customer characteristics
- **Implementation Roadmap**: Created actionable timeline for deploying retention strategies

## Technologies Used

- **Python**: Primary programming language
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models and evaluation
- **XGBoost**: Gradient boosting implementation
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebooks**: Development environment

## Future Enhancements

Potential improvements to this project could include:
- Customer lifetime value (CLV) integration
- Real-time digital engagement monitoring
- Conversational AI for branch-to-digital transition
- Personalized digital adoption pathways
- Churn early warning system
