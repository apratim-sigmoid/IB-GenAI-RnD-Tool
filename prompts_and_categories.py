
overview_prompt = """As an R&D specialist analyzing e-cigarette research, focus on extracting specific, measurable data points and actionable insights.
    Avoid generic safety statements and instead provide precise, quantitative data that can inform product development decisions.
    
    Key areas to emphasize:
    1. Specific chemicals/ingredients and their precise concentrations that show improved safety profiles
    2. Exact device parameters (temperature ranges, wattage levels, coil materials) associated with reduced harmful outputs
    3. Quantitative comparisons between device generations or design features (provide exact percentages or values)
    4. Specific flavor compounds and their safety/satisfaction metrics with exact measurements
    5. Numerical data on user satisfaction correlated with specific product characteristics
    6. Precise operating parameters that optimize nicotine delivery while minimizing harmful constituents
    
    Always prioritize quantitative data and specific technical details that directly enable product improvement."""

categories_to_extract = {
    "Key Findings": [
        "main_conclusions", 
        "statistical_summary.primary_outcomes", 
        "statistical_summary.secondary_outcomes", 
        "novel_findings", 
        "limitations", 
        "generalizability", 
        "future_research_suggestions", 
        "contradictions.conflicts_with_literature",
        "contradictions.internal_contradictions"
    ],
    "Causal Mechanisms": [
        "chemicals_implicated.name",
        "chemicals_implicated.level_detected",
        "chemicals_implicated.effects",
        "chemicals_implicated.evidence_strength",
        "biological_pathways.pathway",
        "biological_pathways.description",
        "device_factors.factor",
        "device_factors.effects",
        "usage_pattern_factors.pattern",
        "usage_pattern_factors.effects"
    ],
    "R&D Insights": [
        "harmful_ingredients.name",
        "harmful_ingredients.health_impact",
        "harmful_ingredients.comparison_to_cigarettes",
        "device_design_implications.feature",
        "device_design_implications.impact",
        "comparative_benefits.vs_traditional_cigarettes.benefit",
        "comparative_benefits.vs_traditional_cigarettes.evidence_strength",
        "potential_innovation_areas.area",
        "operating_parameters.temperature",
        "operating_parameters.wattage"
    ],
    "Study Characteristics": [
        "study_design.primary_type",
        "study_design.secondary_features",
        "study_design.time_periods",
        "sample_characteristics.total_size",
        "sample_characteristics.user_groups",
        "methodology.e_cigarette_specifications.device_types",
        "methodology.e_cigarette_specifications.nicotine_content.concentrations",
        "methodology.data_collection_method"
    ],
    "Health Outcomes": [
        "respiratory_effects.findings.description",
        "cardiovascular_effects.findings.description",
        "oral_health.periodontal_health.description",
        "oral_health.inflammatory_biomarkers.description",
        "neurological_effects.description",
        "psychiatric_effects.description",
        "cancer_risk.description",
        "other_health_outcomes.description"
    ],
}


adverse_events_prompt = """As an R&D specialist analyzing e-cigarette research, focus exclusively on actionable data that identifies specific device factors, ingredients, or operating parameters linked to adverse events.
    Avoid general statements about e-cigarette safety and instead extract precise technical information that can drive product improvements.
    
    Emphasize:
    1. Exact chemical compounds/ingredients at specific concentrations causing adverse effects (e.g., "formaldehyde at >40 μg/puff when device exceeds 240°C")
    2. Precise device characteristics (temperature, wattage, coil material) associated with reduced adverse events
    3. Quantitative comparison data showing which design elements produce fewer adverse events (provide numerical values)
    4. Specific causal pathways between device parameters and biological effects with measured values
    5. Time-dependent adverse event profiles with specific usage patterns (duration, frequency, inhalation technique)
    6. Particular e-liquid formulations showing reduced risk profiles with supporting measurements
    
    Include exact percentages, concentrations, and measurements whenever available."""

adverse_events_categories = {
    "Health Outcomes": [
        "respiratory_effects.measured_outcomes",
        "respiratory_effects.findings.description",
        "respiratory_effects.findings.comparative_results",
        "respiratory_effects.specific_conditions.asthma",
        "respiratory_effects.specific_conditions.copd",
        "respiratory_effects.lung_function_tests.results",
        "cardiovascular_effects.measured_outcomes",
        "cardiovascular_effects.findings.description",
        "cardiovascular_effects.blood_pressure",
        "cardiovascular_effects.heart_rate",
        "neurological_effects.specific_outcomes",
        "psychiatric_effects.specific_outcomes",
        "other_health_outcomes.description"
    ],
    "Self-Reported Effects": [
        "adverse_events.oral_events.sore_dry_mouth.overall_percentage",
        "adverse_events.oral_events.sore_dry_mouth.group_percentages",
        "adverse_events.oral_events.cough.overall_percentage",
        "adverse_events.oral_events.cough.group_percentages",
        "adverse_events.respiratory_events.breathing_difficulties.overall_percentage",
        "adverse_events.respiratory_events.chest_pain.overall_percentage",
        "adverse_events.neurological_events.headache.overall_percentage",
        "adverse_events.neurological_events.dizziness.overall_percentage",
        "adverse_events.cardiovascular_events.heart_palpitation.overall_percentage",
        "adverse_events.total_adverse_events.overall_percentage",
        "adverse_events.systemic_events"
    ],
    "Causal Mechanisms": [
        "chemicals_implicated.name",
        "chemicals_implicated.effects",
        "chemicals_implicated.evidence_strength",
        "biological_pathways.pathway",
        "device_factors.factor",
        "device_factors.effects",
        "usage_pattern_factors.pattern"
    ],
    "Key Findings": [
        "main_conclusions",
        "limitations",
        "novel_findings"
    ]
}


perceived_benefits_prompt = """As an R&D specialist analyzing e-cigarette research, focus exclusively on specific, measurable benefits that can be directly linked to product characteristics.
    Avoid general statements about user satisfaction and instead extract precise technical information that can enhance product appeal and efficacy.
    
    Emphasize:
    1. Exact quantitative improvements in biomarkers or health indicators with specific device types/settings
    2. Precise sensory attributes (with numerical ratings when available) linked to specific flavor compounds 
    3. Detailed user satisfaction metrics correlated with particular product features (provide numerical ratings)
    4. Specific nicotine delivery parameters that optimize satisfaction while minimizing side effects (exact values)
    5. Particular device/e-liquid combinations showing superior user experience with supporting data
    6. Quantitative data on specific product attributes that drive successful transition from combustible cigarettes
    
    Include exact percentages, preference ratings, and measured effects whenever available."""

perceived_benefits_categories = {
    "Self-Reported Effects": [
        "perceived_health_improvements.sensory.smell.overall_percentage",
        "perceived_health_improvements.sensory.smell.group_percentages",
        "perceived_health_improvements.sensory.taste.overall_percentage",
        "perceived_health_improvements.sensory.taste.group_percentages",
        "perceived_health_improvements.sensory.other_sensory_improvements",
        "perceived_health_improvements.physical.breathing.overall_percentage",
        "perceived_health_improvements.physical.physical_status.overall_percentage",
        "perceived_health_improvements.physical.stamina.overall_percentage",
        "perceived_health_improvements.physical.other_physical_improvements",
        "perceived_health_improvements.mental.mood.overall_percentage",
        "perceived_health_improvements.mental.sleep_quality.overall_percentage",
        "perceived_health_improvements.quality_of_life.overall_qol",
        "perceived_health_improvements.quality_of_life.specific_domains"
    ],
    "Behavioral Patterns": [
        "smoking_cessation.success_rates",
        "smoking_cessation.comparison_to_other_methods",
        "smoking_cessation.relapse_rates",
        "reasons_for_use.primary_reasons",
        "reasons_for_use.secondary_reasons",
        "reasons_for_use.demographic_differences"
    ],
    "R&D Insights": [
        "comparative_benefits.vs_traditional_cigarettes.benefit",
        "comparative_benefits.vs_traditional_cigarettes.magnitude",
        "comparative_benefits.vs_other_nicotine_products",
        "consumer_experience_factors.factor",
        "consumer_experience_factors.health_implication",
        "consumer_experience_factors.optimization_suggestion"
    ],
    "Key Findings": [
        "main_conclusions",
        "novel_findings"
    ]
}


oral_health_prompt = """As an R&D specialist analyzing e-cigarette research on oral health, focus exclusively on specific product parameters and ingredients that impact oral health outcomes.
    Avoid general statements about oral health impacts and instead extract precise technical information that can directly inform product formulation and design.
    
    Emphasize:
    1. Specific e-liquid ingredients and their measured effects on oral microbiome (with CFU counts or other metrics)
    2. Exact pH levels of various e-liquids and their impact on dental erosion (with numerical measurements)
    3. Precise temperature ranges associated with reduced oral irritation (with exact values)
    4. Particular flavoring compounds linked to improved or worsened oral health outcomes (with measured effects)
    5. Specific device design elements that minimize oral tissue exposure to harmful aerosols (with quantified reduction)
    6. Comparative data on oral biomarkers between specific product types/generations (with exact values)
    
    Include specific chemical names, concentrations, and measured biological responses whenever available."""

oral_health_categories = {
    "Health Outcomes": [
        "oral_health.periodontal_health.description",
        "oral_health.periodontal_health.measurements",
        "oral_health.periodontal_health.significance",
        "oral_health.periodontal_health.comparison.effect",
        "oral_health.caries_risk.description",
        "oral_health.caries_risk.measurements",
        "oral_health.oral_mucosal_changes.description",
        "oral_health.oral_mucosal_changes.types_of_lesions",
        "oral_health.inflammatory_biomarkers.description",
        "oral_health.inflammatory_biomarkers.biomarkers_studied",
        "oral_health.other_oral_effects.description"
    ],
    "Self-Reported Effects": [
        "adverse_events.oral_events.sore_dry_mouth.overall_percentage",
        "adverse_events.oral_events.sore_dry_mouth.time_course",
        "adverse_events.oral_events.mouth_tongue_sores.overall_percentage",
        "adverse_events.oral_events.gingivitis.overall_percentage",
        "adverse_events.oral_events.other_oral_events.event",
        "adverse_events.oral_events.other_oral_events.percentage",
        "adverse_events.oral_events.cough.overall_percentage"
    ],
    "Causal Mechanisms": [
        "chemicals_implicated.name",
        "chemicals_implicated.effects", 
        "biological_pathways.pathway",
        "biological_pathways.description"
    ],
    "Key Findings": [
        "main_conclusions", 
        "novel_findings"
    ]
}


respiratory_prompt = """As an R&D specialist analyzing e-cigarette research on respiratory health, focus exclusively on specific product parameters and ingredients that impact respiratory health outcomes.
    Avoid general statements about respiratory health and instead extract precise technical information that can directly improve product safety profiles.
    
    Emphasize:
    1. Specific aerosol particle sizes from different device types and their measured deposition patterns
    2. Exact chemical compounds at specific concentrations linked to respiratory irritation
    3. Precise temperature/power settings associated with reduced respiratory effects (with numerical values)
    4. Particular e-liquid formulations showing improved respiratory safety profiles (with measured data)
    5. Specific device design elements that demonstrably filter or reduce harmful respiratory exposures
    6. Comparative respiratory biomarker data between specific product designs (with exact measurements)
    
    Include specific quantitative data on aerosol physics, chemical composition, and physiological responses whenever available."""

respiratory_categories = {
    "Health Outcomes": [
        "respiratory_effects.measured_outcomes",
        "respiratory_effects.findings.description",
        "respiratory_effects.findings.comparative_results",
        "respiratory_effects.specific_conditions.asthma",
        "respiratory_effects.specific_conditions.copd",
        "respiratory_effects.specific_conditions.wheezing",
        "respiratory_effects.specific_conditions.other_conditions",
        "respiratory_effects.biomarkers",
        "respiratory_effects.lung_function_tests.tests_performed",
        "respiratory_effects.lung_function_tests.results"
    ],
    "Self-Reported Effects": [
        "adverse_events.respiratory_events.breathing_difficulties.overall_percentage",
        "adverse_events.respiratory_events.breathing_difficulties.group_percentages",
        "adverse_events.respiratory_events.chest_pain.overall_percentage",
        "adverse_events.respiratory_events.chest_pain.group_percentages",
        "adverse_events.respiratory_events.other_respiratory_events",
        "adverse_events.oral_events.cough.overall_percentage",
        "adverse_events.oral_events.cough.time_course"
    ],
    "Causal Mechanisms": [
        "chemicals_implicated.name",
        "chemicals_implicated.effects",
        "biological_pathways.pathway",
        "biological_pathways.description"
    ],
    "Key Findings": [
        "main_conclusions", 
        "novel_findings"
    ]
}


cardiovascular_prompt = """As an R&D specialist analyzing e-cigarette research on cardiovascular health, focus exclusively on specific product parameters and ingredients that impact cardiovascular health metrics.
    Avoid general statements about cardiovascular risks and instead extract precise technical information that can directly inform product design and formulation.
    
    Emphasize:
    1. Specific nicotine delivery patterns and their measured effects on heart rate/blood pressure (with exact values)
    2. Exact chemical constituents linked to vascular effects with their concentrations
    3. Precise operating parameters associated with minimized cardiovascular impact (with numerical data)
    4. Particular device design elements showing improved cardiovascular safety profiles
    5. Specific e-liquid formulations with measured reduced impact on endothelial function
    6. Comparative cardiac biomarker data between specific product types (with exact measurements)
    
    Include exact measurements, concentration ranges, and physiological response data whenever available."""

cardiovascular_categories = {
    "Health Outcomes": [
        "cardiovascular_effects.measured_outcomes",
        "cardiovascular_effects.findings.description",
        "cardiovascular_effects.findings.comparative_results",
        "cardiovascular_effects.blood_pressure",
        "cardiovascular_effects.heart_rate",
        "cardiovascular_effects.biomarkers"
    ],
    "Self-Reported Effects": [
        "adverse_events.cardiovascular_events.heart_palpitation.overall_percentage",
        "adverse_events.cardiovascular_events.heart_palpitation.group_percentages",
        "adverse_events.cardiovascular_events.other_cardiovascular_events"
    ],
    "Causal Mechanisms": [
        "chemicals_implicated.name",
        "chemicals_implicated.effects",
        "biological_pathways.pathway",
        "biological_pathways.description"
    ],
    "Key Findings": [
        "main_conclusions", 
        "novel_findings"
    ]
}


research_trends_prompt = """As an R&D specialist analyzing e-cigarette research trends, focus exclusively on emerging R&D directions and quantifiable product trends.
    Avoid general statements about industry evolution and instead extract precise technical information that can inform product development strategy.
    
    Emphasize:
    1. Specific next-generation device technologies with measured performance improvements
    2. Exact new e-liquid formulations showing enhanced safety/satisfaction profiles (with data)
    3. Precise shifting consumer preferences with numerical market data
    4. Particular emerging testing methodologies that provide more accurate product assessment
    5. Specific regulatory trends that will impact product development (with implementation timelines)
    6. Quantifiable trends in competing products with exact market share or growth figures
    
    Include specific technical innovations, measurable market shifts, and emerging research methodologies whenever available."""

research_trends_categories = {
    "Study Characteristics": [
        "study_design.primary_type",
        "study_design.secondary_features", 
        "study_design.time_periods",
        "methodology.data_collection_method",
        "methodology.e_cigarette_specifications.device_types",
        "methodology.e_cigarette_specifications.generation",
        "methodology.e_cigarette_specifications.nicotine_content.concentrations",
        "methodology.e_cigarette_specifications.e_liquid_types",
        "methodology.e_cigarette_specifications.flavors_studied"
    ],
    "Key Findings": [
        "future_research_suggestions",
        "novel_findings"
    ],
    "Market Trends": [
        "product_characteristics.device_evolution",
        "product_characteristics.e_liquid_trends",
        "product_characteristics.nicotine_concentration_trends",
        "product_characteristics.price_trends",
        "regulatory_impacts.regulation_effects",
        "regulatory_impacts.policy_recommendations"
    ],
    "Behavioral Patterns": [
        "usage_patterns.transitions.description",
        "usage_patterns.transitions.from_smoking_to_vaping",
        "usage_patterns.transitions.from_vaping_to_smoking",
        "usage_patterns.transitions.dual_use_patterns",
        "product_preferences.device_preferences.most_popular_devices",
        "product_preferences.flavor_preferences.most_popular_flavors",
        "product_preferences.nicotine_preferences.most_common_concentrations"
    ]
}


contradictions_prompt = """As an R&D specialist analyzing conflicting e-cigarette research, focus exclusively on specific technical disagreements in the literature that impact product development decisions.
    Avoid general statements about research limitations and instead extract precise information about conflicting findings relevant to product design.
    
    Emphasize:
    1. Exact contradictory findings about specific ingredients/concentrations and their effects
    2. Precise conflicting data about optimal operating parameters (temperature, wattage) with specific values
    3. Particular methodological differences explaining contradictory safety assessments of specific components
    4. Specific disagreements about aerosol chemistry under different conditions (with measurements)
    5. Quantitative discrepancies in biomarker responses to particular product characteristics
    6. Contradictory consumer preference data about specific product features (with numerical values)
    
    Include specific numerical data points from conflicting studies and identify potential reasons for discrepancies."""
    
contradictions_categories = {
    "Key Findings": [
        "contradictions.conflicts_with_literature",
        "contradictions.internal_contradictions",
        "generalizability",
        "limitations"
    ],
    "Bias Assessment": [
        "conflicts_of_interest.description",
        "conflicts_of_interest.industry_affiliations",
        "conflicts_of_interest.transparency",
        "methodological_concerns",
        "overall_quality_assessment"
    ],
    "Causal Mechanisms": [
        "chemicals_implicated.name",
        "chemicals_implicated.evidence_strength",
        "biological_pathways.pathway",
        "biological_pathways.evidence_strength"
    ],
    "R&D Insights": [
        "comparative_benefits.vs_traditional_cigarettes.benefit",
        "comparative_benefits.vs_traditional_cigarettes.evidence_strength",
        "harmful_ingredients.name",
        "harmful_ingredients.health_impact",
        "harmful_ingredients.comparison_to_cigarettes",
        "harmful_ingredients.evidence_strength"
    ]
}


bias_prompt = """As an R&D specialist analyzing methodological issues in e-cigarette research, focus exclusively on methodological issues that could distort understanding of specific product characteristics.
    Avoid general comments about research quality and instead extract precise information about testing and measurement methods relevant to product development.
    
    Emphasize:
    1. Specific product testing protocols that produce particularly reliable/unreliable data
    2. Exact measurement techniques for aerosol constituents with identified limitations/advantages
    3. Particular study designs that provide the most actionable product development insights
    4. Specific control variables that significantly impact assessment of product safety/satisfaction
    5. Methodological best practices for evaluating specific aspects of e-cigarette performance
    6. Measurement biases affecting evaluation of particular device features or e-liquid components
    
    Include specific analytical methods, instrument specifications, and experimental design considerations whenever available."""
    
bias_categories = {
    "Bias Assessment": [
        "selection_bias",
        "measurement_bias",
        "confounding_factors",
        "attrition_bias",
        "reporting_bias",
        "conflicts_of_interest.description",
        "conflicts_of_interest.industry_affiliations",
        "conflicts_of_interest.transparency",
        "methodological_concerns",
        "overall_quality_assessment"
    ],
    "Meta Data": [
        "funding_source.type",
        "funding_source.specific_entities",
        "funding_source.disclosure_statement"
    ],
    "Key Findings": [
        "limitations",
        "generalizability"
    ],
    "Study Characteristics": [
        "statistical_methods.adjustment_factors",
        "methodology.control_variables",
        "methodology.inclusion_criteria",
        "methodology.exclusion_criteria"
    ]
}


publication_prompt = """As an R&D specialist analyzing e-cigarette research quality, focus exclusively on identifying the most credible and technically rigorous product research.
    Avoid general assessment of publication patterns and instead extract precise information about the most reliable sources of technical product data.
    
    Emphasize:
    1. Specific publications/research groups producing the most methodologically sound product assessments
    2. Exact analytical methods that provide most reliable data on product characteristics
    3. Particular study designs yielding most actionable product improvement insights
    4. Specific technical expertise patterns across research institutions
    5. Most rigorous comparative studies between product types with detailed methodologies
    6. Cutting-edge analytical techniques being applied to product assessment
    
    Include specific research institutions, analytical methodologies, and citation metrics for the most technically reliable research."""

publication_categories = {
    "Meta Data": [
        "publication_type",
        "journal",
        "citation_info",
        "publication_year",
        "country_of_study",
        "authors",
        "funding_source.type",
        "funding_source.specific_entities"
    ],
    "Study Characteristics": [
        "sample_characteristics.total_size",
        "study_design.primary_type",
        "study_design.secondary_features",
        "statistical_methods.primary_analyses",
        "statistical_methods.secondary_analyses"
    ],
    "Key Findings": [
        "statistical_summary.primary_outcomes",
        "statistical_summary.secondary_outcomes",
        "main_conclusions",
        "novel_findings"
    ],
    "Bias Assessment": [
        "overall_quality_assessment",
        "conflicts_of_interest.description",
        "conflicts_of_interest.industry_affiliations"
    ]
}