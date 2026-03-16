import pandas as pd

def count_pipe(s):
    if not s or str(s).strip() == '': return 0
    return len([x for x in str(s).split('|') if x.strip()])

def main():
    csv_path = r'llm-as-ontology-server\output\gemini\run_002\step3_accuracy\comparison_set1_set2_vs_ground_truth.csv'
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna('')
    except Exception as e:
        print(f"Error: {e}")
        return

    questionnaire_data = []
    categories = ['parents', 'grandparents', 'children', 'siblings']
    
    # Validation options to be used in the form
    options = "Valid (Clinical) | Valid (Synonym) | Error (Incorrect) | Error (Irrelevant)"

    for _, row in df.iterrows():
        concept = row['concept_name']
        
        # Check all 4 categories across both sets
        for cat in categories:
            # We combine added terms from set1 and set2 for the researcher to validate
            added_set1 = set([x.strip() for x in str(row[f'set1_{cat}_added']).split('|') if x.strip()])
            added_set2 = set([x.strip() for x in str(row[f'set2_{cat}_added']).split('|') if x.strip()])
            all_added = added_set1.union(added_set2)
            
            if all_added:
                added_str = " | ".join(sorted(list(all_added)))
                questionnaire_data.append({
                    'Concept Name': concept,
                    'Relationship Type': cat.capitalize(),
                    'Added Terms to Validate': added_str,
                    'Suggested Validation Options': options,
                    'Question': f"For the concept '{concept}', the LLM added the following {cat} terms. Are they clinically valid?"
                })

    if questionnaire_data:
        out_df = pd.DataFrame(questionnaire_data)
        output_path = 'medical_researcher_questionnaire_full.csv'
        out_df.to_csv(output_path, index=False)
        print(f"Successfully generated questionnaire for {len(out_df)} concept-category pairs.")
        print(f"Saved to: {output_path}")
    else:
        print("No added terms found to generate questions.")

if __name__ == "__main__":
    main()
