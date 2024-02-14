import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

# Load data from files
di_sy_res = pd.read_csv('Di_sy_res.csv')
diseases = pd.read_csv('diseases.csv')
symptoms = pd.read_csv('Symptoms.csv')

# Function to get diseases related to a specific symptom
def get_diseases_for_symptom(symptom_id):
    related_diseases = di_sy_res[di_sy_res['Symptom'] == symptom_id]['Disease'].unique()
    return related_diseases

# Function to get symptoms for a specific disease
def get_symptoms_for_disease(disease_id):
    disease_symptoms = di_sy_res[di_sy_res['Disease'] == disease_id]['Symptom']
    symptom_names = symptoms[symptoms['primary'].isin(disease_symptoms)]['name'].tolist()
    return symptom_names

def get_symptoms_for_disease(disease_name):
    disease_symptoms = di_sy_res[di_sy_res['Disease'] == disease_name]['Symptom']
    symptom_names = symptoms[symptoms['primary'].isin(disease_symptoms)]['name'].tolist()
    return symptom_names

# Function to get diseases related to a list of symptoms
def get_diseases_for_symptoms(symptom_list):
    related_diseases = set()
    for symptom in symptom_list:
        related_diseases.update(get_diseases_for_symptom(symptom))
    return list(related_diseases)

# Function to get top N most frequent symptoms for a specific disease
def get_top_n_symptoms_for_disease(disease_id, n=5):
    disease_symptoms = di_sy_res[di_sy_res['Disease'] == disease_id]['Symptom']
    
    top_n_symptoms = disease_symptoms.value_counts().index[:n]
    return top_n_symptoms

# Function to get diseases related to a list of symptoms
def get_diseases_for_symptoms(symptom_list):
    related_diseases = set()
    for symptom in symptom_list:
        related_diseases.update(get_diseases_for_symptom(symptom))
    return list(related_diseases)

# Function to generate CSV file for a specific disease and its related symptoms
def generate_csv_for_disease(disease_id, output_file):
    most_frequent_symptoms = get_top_n_symptoms_for_disease(disease_id)
    related_diseases = get_diseases_for_symptoms(most_frequent_symptoms)
    
    # Create a DataFrame with disease and related symptoms
    df_result = pd.DataFrame({'Disease': [disease_id]*len(most_frequent_symptoms), 'Symptom': most_frequent_symptoms})
    
    # Save the result to a CSV file
    df_result.to_csv(output_file, index=False)


# Function to get disease_id for a specific disease name
def get_disease_id(disease_name):
    matching_diseases = diseases[diseases['name'] == disease_name]
    if not matching_diseases.empty:
        return matching_diseases['do_id'].values[0]
    return None

# Streamlit App
st.title("Disease Prediction and Symptom Analysis")

# Ask user for three input symptoms
symptom_questions = ["What brings you here today?", "What are your symptoms?", ""]
user_selected_symptoms = []

for question in symptom_questions:
    symptom = st.text_input(question)
    user_selected_symptoms.append(symptom)

# Display the selected symptoms
st.subheader("Your Symptoms:")
st.write(", ".join(user_selected_symptoms))


# Question 1: When did your symptoms start? (In Days)
symptoms_start = st.number_input("1. When did your symptoms start? (In Days)", min_value=0)

# Question 2: Have your symptoms gotten better or worse?
symptoms_condition = st.radio("2. Have your symptoms gotten better or worse?", options=["Better", "Worse"])


score = st.slider('How much pain you are feeling (SCORE: 1 to 100)', min_value=0, max_value=100, value=50)

# Question 3: Have you had any procedures or major illnesses in the past 12 months?
past_illnesses = st.radio("3. Have you had any procedures or major illnesses in the past 12 months?", options=["Yes", "No"])
if past_illnesses == "Yes":
    illnesses_description = st.text_area("   Illnesses Description")

# Question 4: What medications do you take?
medications = st.text_area("4. What prescription medications, over-the-counter medications, vitamins, and supplements do you take? Which ones have you been on in the past? (Descriptive)")

# Question 5: What allergies do you have?
allergies = st.radio("5. Do you have any allergies?", options=["Yes", "No"])
if allergies == "Yes":
    allergies_description = st.text_area("   Allergies Description")

# Question 6: Have you served in the military?
military_service = st.radio("6. Have you served in the military?", options=["Yes", "No"])

# Question 7: Are you sexually active?
sexual_activity = st.radio("7. Are you sexually active?", options=["Active", "Inactive"])

# substance_use = st.checkbox("8. Do you use any kind of tobacco, illicit drugs, or alcohol?")
# if substance_use:
#     tobacco = st.checkbox("   Tobacco")
#     illicit_drugs = st.checkbox("   Illicit Drugs")
#     alcohol = st.checkbox("   Alcohol")




# Find diseases matching at least 2 out of 3 symptoms
matching_diseases = []
for disease_id in diseases['do_id']:
    disease_symptoms = get_symptoms_for_disease(disease_id)
    matching_symptoms = set(user_selected_symptoms) & set(disease_symptoms)
    
    if len(matching_symptoms) >= 2:
        matching_diseases.append({'Disease ID': disease_id,
                                  'Disease Name': diseases[diseases['do_id'] == disease_id]['name'].values[0],
                                  'Matching Symptoms': ', '.join(matching_symptoms)})

# Create a DataFrame for matching diseases
matching_diseases_df = pd.DataFrame(matching_diseases)

# Display the matching diseases table
st.subheader("Diseases Matching at Least 2 out of 3 Symptoms:")
st.table(matching_diseases_df)


# Show all symptoms for each matched disease
st.subheader("Symptoms for Matched Diseases:")
matched_disease_symptoms = []

for _, row in matching_diseases_df.iterrows():
    disease_id = row['Disease ID']
    disease_name = row['Disease Name']
    matched_symptoms = get_symptoms_for_disease(disease_id)

    matched_disease_symptoms.append({'Disease Name': disease_name,
                                     'Disease ID': disease_id,
                                     'Symptoms': ', '.join(matched_symptoms)})

# Create a DataFrame for matched disease symptoms
matched_disease_symptoms_df = pd.DataFrame(matched_disease_symptoms)

# Display the table
matched_disease_symptoms_table = st.table(matched_disease_symptoms_df)


# Display the union of selected and matched symptoms in tabular form
union_symptoms = set(user_selected_symptoms)
# union_symptoms = set(union_symptoms)

for _, row in matching_diseases_df.iterrows():
    disease_id = row['Disease ID']
    matched_symptoms = set(get_symptoms_for_disease(disease_id))
    union_symptoms |= matched_symptoms


num_symptoms_to_show = 5
limited_union_symptoms = list(union_symptoms)[:num_symptoms_to_show]

# Create a DataFrame with the limited union of selected and matched symptoms
limited_union_df = pd.DataFrame({'Union of Selected and Matched Symptoms': limited_union_symptoms})

# Display the DataFrame
st.subheader(f"Top {num_symptoms_to_show} Union of Selected and Matched Symptoms:")
st.table(limited_union_df)

# Create a DataFrame with the union of selected and matched symptoms
# union_df = pd.DataFrame({'Union of Selected and Matched Symptoms': list(union_symptoms)})

# # Display the DataFrame
# st.subheader("Union of Selected and Matched Symptoms:")
# st.table(union_df)


# Ask user about family disease
family_disease_response = st.radio("Do you have a family/genetic disease?", ["Yes", "No"])

family_disease_symptoms =[]
if family_disease_response == "Yes":
    family_disease_name = st.text_input("Name the Family/Genetic Disease:")
    family_disease_id = get_disease_id(family_disease_name)

    if family_disease_id:
        family_disease_symptoms = get_symptoms_for_disease(family_disease_id)

        # Display symptoms of the family disease
        if family_disease_symptoms:
            st.subheader(f"Symptoms of {family_disease_name} (Disease ID: {family_disease_id}):")

            # Create a DataFrame for the symptoms
            family_symptoms_df = pd.DataFrame({
                "Disease Name": [family_disease_name],
                "Symptoms": [', '.join(family_disease_symptoms)]
            })

            # Append the family symptoms to the matched_disease_symptoms_df
            matched_disease_symptoms_df = pd.concat([matched_disease_symptoms_df, family_symptoms_df], ignore_index=True)

            # Display the updated symptoms table
            matched_disease_symptoms_table.empty()
            matched_disease_symptoms_table.table(matched_disease_symptoms_df)

            # Display the symptoms table
            st.table(family_symptoms_df)
        else:
            st.write(f"No symptoms available for the specified family disease: {family_disease_name}")

# Store the union of family symptoms and selected/matched symptoms
all_symptoms_union = set(family_disease_symptoms) | union_symptoms


# Create a DataFrame for the union of symptoms
all_symptoms_union_df = pd.DataFrame({
    "Symptoms Union": [', '.join(all_symptoms_union)]
})
all_symptoms_union_df = set(all_symptoms_union_df)

# Display the union of symptoms
st.subheader("Union of Family Symptoms, Selected Symptoms, and Matched Symptoms:")
all_symptoms_union_df_table = st.table(all_symptoms_union_df)


def train_model_on_symptoms(symptoms_list):
    # Create a DataFrame with symptoms
    df = pd.DataFrame(symptoms_list, columns=['Symptom'])

    # Check if there are enough samples for training
    if len(df) < 2:
        st.warning("Insufficient data for training. Please select more symptoms.")
        return None, None, None

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # Use TfidfVectorizer to convert symptoms into numerical features
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data['Symptom'])
    X_test = vectorizer.transform(test_data['Symptom'])

    # Use MultiLabelBinarizer to convert multiple diseases into binary labels
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_data['Symptom'].apply(lambda x: [x]))
    y_test = mlb.transform(test_data['Symptom'].apply(lambda x: [x]))

    # Use a classification algorithm, such as Multinomial Naive Bayes
    model = OneVsRestClassifier(MultinomialNB())
    model.fit(X_train, y_train)

    return model, vectorizer, mlb

# Convert set to list
symptoms_list = list(all_symptoms_union)
trained_symptoms_model, symptoms_vectorizer, mlb_symptoms = train_model_on_symptoms(symptoms_list)
# Create a DataFrame with symptoms data
symptoms_df = pd.DataFrame(symptoms_list, columns=['Symptom'])

# Example of using the trained model for predictions
def predict_top_n_symptoms(model, vectorizer, mlb, input_symptoms, n=5):

    # Transform input symptoms using the same vectorizer
    input_symptoms_vectorized = vectorizer.transform(input_symptoms)

    # Make predictions for the input symptoms
    predicted_symptoms_prob = model.predict_proba(input_symptoms_vectorized)

    # Get the indices of top N predicted symptoms
    top_n_symptoms_indices = np.argsort(predicted_symptoms_prob[0])[::-1][:n]

    # Retrieve the corresponding symptom labels
    top_n_symptoms = mlb.classes_[top_n_symptoms_indices]

    return top_n_symptoms, predicted_symptoms_prob

# Predict top symptoms based on the all_symptoms_union
top_predicted_symptoms, predicted_symptoms_prob = predict_top_n_symptoms(
    trained_symptoms_model, symptoms_vectorizer, mlb_symptoms, symptoms_list
)

# Display the top 3 predicted symptoms and their probabilities
st.subheader("Top Predicted Symptoms:")
for symptom_label, probability in zip(top_predicted_symptoms, predicted_symptoms_prob[0][:len(top_predicted_symptoms)]):
    st.write(f"{symptom_label}")


# Display the top predicted symptoms
st.subheader("Check the symptoms you are experiencing")
checked_symptoms = st.multiselect("", top_predicted_symptoms)

# Store the checked and unchecked symptoms
unchecked_symptoms = [symptom for symptom in top_predicted_symptoms if symptom not in checked_symptoms]


disease_symptoms = list(set(disease_symptoms) - set(unchecked_symptoms))
all_symptoms_union -= set(unchecked_symptoms)



# Remove unchecked symptoms from the union of symptoms
union_symptoms -= set(unchecked_symptoms)

union_symptoms = set(union_symptoms)
# Initialize the matched_disease_symptoms_dict with scores set to zero
matched_disease_symptoms_dict = {row['Disease Name']: 0 for _, row in matched_disease_symptoms_df.iterrows()}

# Check and update scores based on checked symptoms
for _, row in matched_disease_symptoms_df.iterrows():
    disease_name = row['Disease Name']
    disease_symptoms = get_symptoms_for_disease(row['Disease ID'])

    # Remove unchecked symptoms from individual remaining diseases
    disease_symptoms = list(set(disease_symptoms) - set(unchecked_symptoms))

    # Only consider checked symptoms for updating scores
    for checked_symptom in checked_symptoms:
        if checked_symptom in disease_symptoms:
            matched_disease_symptoms_dict[disease_name] += 1
        else:
            matched_disease_symptoms_dict[disease_name] -= 1

# Update the union of symptoms based on the removed symptoms
union_symptoms = set()
for _, row in matched_disease_symptoms_df.iterrows():
    union_symptoms |= set(get_symptoms_for_disease(row['Disease ID']))



# Remove unchecked symptoms from the model training data
all_symptoms_union -= set(unchecked_symptoms)
all_symptoms_union = set(all_symptoms_union)
# Display the scores in a table
st.subheader("Scores for Matched Diseases:")
scores_data = [{'Disease Name': disease_name, 'Score': score} for disease_name, score in matched_disease_symptoms_dict.items()]
scores_df = pd.DataFrame(scores_data)
st.table(scores_df)

# Remove diseases with scores less than -2 or greater than 2
filtered_matched_disease_symptoms = {disease_name: score for disease_name, score in matched_disease_symptoms_dict.items() if score > -2}
filtered_matched_diseases_symptoms = [disease_name for disease_name, score in filtered_matched_disease_symptoms.items()]

# Update matched_disease_symptoms_dict and matched_diseases_symptoms
matched_disease_symptoms_dict = filtered_matched_disease_symptoms
matched_diseases_symptoms = filtered_matched_diseases_symptoms

# Display the updated scores and diseases
st.subheader("Updated Scores for Matched Diseases:")
updated_scores_data = [{'Disease Name': disease_name, 'Score': score} for disease_name, score in matched_disease_symptoms_dict.items()]
updated_scores_df = pd.DataFrame(updated_scores_data)

# Update the matched_disease_symptoms_df
matched_disease_symptoms_df = pd.DataFrame(matched_disease_symptoms)
    
# Filter matched_disease_symptoms_df based on diseases in updated_scores_df
matched_disease_symptoms_df_filtered = matched_disease_symptoms_df[matched_disease_symptoms_df['Disease Name'].isin(updated_scores_df['Disease Name'])]

# Clear the existing table and display the updated one
matched_disease_symptoms_table.empty()
matched_disease_symptoms_table.table(matched_disease_symptoms_df_filtered)
st.table(updated_scores_df)

# Update the union of symptoms based on the updated diseases table
union_symptoms = set()
for _, row in matched_disease_symptoms_df_filtered.iterrows():
     union_symptoms |= set(get_symptoms_for_disease(row['Disease ID']))

# Create a DataFrame for the updated union of symptoms
updated_union_df = pd.DataFrame([{'Symptoms Union': ', '.join(union_symptoms)}])
updated_union_df = set(updated_union_df)

# Display the DataFrame
union_df_table = st.empty()

# Clear the existing table and display the updated union of symptoms
union_df_table.empty()
union_df_table.table(updated_union_df)

# Store the union of family symptoms and selected/matched symptoms
all_symptoms_union = set(family_disease_symptoms) | union_symptoms
all_symptoms_union -=set(unchecked_symptoms)

# Create a DataFrame for the union of symptoms
all_symptoms_union_df = pd.DataFrame({
    "Symptoms Union": [', '.join(all_symptoms_union)]
})

all_symptoms_union_df = set(all_symptoms_union_df)

all_symptoms_union_df_table.empty()
all_symptoms_union_df_table.table(all_symptoms_union_df)



#---------------------------------------------------------------
# Accumulate symptoms from updated diseases
updated_diseases_symptoms = set()
for _, row in matched_disease_symptoms_df_filtered.iterrows():
    disease_id = row['Disease ID']
    disease_symptoms = set(get_symptoms_for_disease(disease_id))
    updated_diseases_symptoms |= set(disease_symptoms)

# Merge symptoms with commas
merged_symptoms = ', '.join(updated_diseases_symptoms)
# Create a DataFrame with the merged symptoms
merged_symptoms_df = pd.DataFrame({"Symptoms": [merged_symptoms]})

# Display the DataFrame
st.subheader("Union of Updated Diseases' Symptoms:")
st.table(merged_symptoms_df)



#-----------------------------------------------------------------------------
# Predict top symptoms based on updated diseases' symptoms
top_predicted_symptoms_new, predicted_symptoms_prob = predict_top_n_symptoms(
    trained_symptoms_model, symptoms_vectorizer, mlb_symptoms, merged_symptoms_df['Symptoms'], n=3
)

#---------------------------------------------------------------------updating----------------------------------------------------------

# # Display the top predicted symptoms
# st.subheader("Top Predicted Symptoms based on Updated Diseases' Symptoms:")
# for symptom_label, probability in zip(top_predicted_symptoms_new, predicted_symptoms_prob[0][:len(top_predicted_symptoms_new)]):
#     st.write(f"{symptom_label}")


# # Display the top predicted symptoms
# st.subheader("Check the symptoms you are experiencing")
# checked_symptoms = st.multiselect("", top_predicted_symptoms_new)

#---------------------------------------------------------------------updating----------------------------------------------------------
# Display the top predicted symptoms
st.subheader("Top Predicted Symptoms based on Updated Diseases' Symptoms:")
for symptom_label, probability in zip(top_predicted_symptoms_new, predicted_symptoms_prob[0][:len(top_predicted_symptoms_new)]):
    st.write(f"{symptom_label}")

# Display the top predicted symptoms
st.subheader("Check the symptoms you are experiencing")
checked_symptoms = st.multiselect("", top_predicted_symptoms_new)

# Check and update scores based on checked symptoms
for disease_name, row in updated_scores_df.iterrows():
    disease_symptoms = get_symptoms_for_disease(disease_name)  # Assuming you have a function to get symptoms for a disease

    # Only consider checked symptoms for updating scores
    for checked_symptom in checked_symptoms:
        if checked_symptom in disease_symptoms:
            updated_scores_df.at[disease_name, 'Score'] += 1
        else:
            updated_scores_df.at[disease_name, 'Score'] -= 1

# Display the updated scores in a table
st.subheader("Updated Scores for Matched Diseases:")
st.table(updated_scores_df)

# Remove diseases with scores less than -2 or greater than 2
filtered_updated_scores_df = updated_scores_df[(updated_scores_df['Score'] >= -2) & (updated_scores_df['Score'] <= 2)]

# Display the filtered updated scores
st.subheader("Filtered Updated Scores for Matched Diseases:")
st.table(filtered_updated_scores_df)
