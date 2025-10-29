#!/usr/bin/env python3
"""
PostNL AI API - Gemini Integration for AG Grid AI Toolkit
Converts natural language queries to AG Grid state changes
"""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Vercel dashboard

# Configure Gemini
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyA7BfIP9ZNzhy6cYctnJMA0jmdfVbtesn4')
genai.configure(api_key=GEMINI_API_KEY)

# Use Gemini 2.0 Flash (FREE tier, fast, accurate)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'gemini-2.0-flash-exp'})

@app.route('/ai/query', methods=['POST'])
def ai_query():
    """
    Process natural language query and return AG Grid state changes

    Request Body:
    {
        "query": "Show me all TRUCK parcels from Jency",
        "schema": {...},  // AG Grid structured schema
        "currentState": {...}  // Current grid state
    }

    Response:
    {
        "success": true,
        "newState": {...},  // New grid state to apply
        "explanation": "Filtered to TRUCK mode and sender 'jency'"
    }
    """
    try:
        data = request.json
        user_query = data.get('query', '')
        grid_schema = data.get('schema', {})
        current_state = data.get('currentState', {})
        tab = data.get('tab', 'parcels')  # emails, parcels, shipments, attachments

        if not user_query:
            return jsonify({'success': False, 'error': 'No query provided'}), 400

        # Build system prompt with context
        system_prompt = f"""You are an AI assistant that converts natural language queries into AG Grid state changes.

CURRENT TAB: {tab}

AVAILABLE COLUMNS AND THEIR MEANINGS:
Parcels tab:
- mode_of_transport: "TRUCK" or "AIR" (shipping method)
- tracking_number: Parcel tracking ID
- trip_number: Shipment/trip identifier
- scac_code: Carrier code
- consignee_name: Company receiving the parcel
- weight: Parcel weight in kilograms
- commercial_value: Value in USD
- origin_country: Country of origin
- dest_country: Destination country
- source_file: Excel file this parcel came from

Emails tab:
- from_email: Sender email address (e.g., jency.varghese@spring-gds.com, thomas@rh-brokerage.com)
- subject: Email subject line
- received_date: When email was received
- type: "TRUCK" (DSLQ emails), "AIR" (ORD/USA emails), or generic

Shipments tab:
- trip_number: Trip/AWB identifier
- mode: "TRUCK" or "AIR"
- parcel_count: Number of parcels in shipment
- total_weight: Total weight of shipment
- source_file: Source file name

Attachments tab:
- filename: Name of attached file
- from_email: Sender who sent the attachment
- size: File size in bytes
- received_date: Date received

GRID SCHEMA:
{json.dumps(grid_schema, indent=2)}

CURRENT STATE:
{json.dumps(current_state, indent=2)}

USER QUERY: "{user_query}"

Your task:
1. Understand what the user wants to do with the grid
2. Generate a NEW grid state that accomplishes their goal
3. Return ONLY valid JSON matching the schema structure

Common operations:
- Filtering: Use filterModel to filter by column values
- Sorting: Use sortModel to sort columns
- Grouping: Use rowGroupColumns and groupKeys
- Column visibility: Use columnState to hide/show columns

Examples:
Query: "Show me all TRUCK parcels"
Response: {{"filterModel": {{"mode_of_transport": {{"filterType": "text", "type": "equals", "filter": "TRUCK"}}}}}}

Query: "Sort by weight highest first"
Response: {{"sortModel": [{{"colId": "weight", "sort": "desc"}}]}}

Query: "Show emails from Jency"
Response: {{"filterModel": {{"from_email": {{"filterType": "text", "type": "contains", "filter": "jency"}}}}}}

Query: "Group by mode of transport"
Response: {{"rowGroupColumns": ["mode_of_transport"]}}

Query: "Filter to October parcels"
Response: {{"filterModel": {{"source_file": {{"filterType": "text", "type": "contains", "filter": "OCT"}}}}}}

Now generate the new state for the user's query. Return ONLY valid JSON.
"""

        # Call Gemini with structured output
        response = model.generate_content(
            system_prompt,
            generation_config={
                'temperature': 0.1,  # Low temperature for deterministic output
                'response_mime_type': 'application/json'
            }
        )

        # Parse response
        new_state = json.loads(response.text)

        # Generate explanation
        explanation = generate_explanation(user_query, new_state)

        return jsonify({
            'success': True,
            'newState': new_state,
            'explanation': explanation
        })

    except json.JSONDecodeError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid JSON response from AI: {str(e)}'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def generate_explanation(query, state):
    """Generate human-readable explanation of state changes"""
    explanations = []

    # Check filters
    if 'filterModel' in state and state['filterModel']:
        filters = state['filterModel']
        for col, filter_def in filters.items():
            filter_value = filter_def.get('filter', '')
            filter_type = filter_def.get('type', 'contains')
            explanations.append(f"Filtered {col} {filter_type} '{filter_value}'")

    # Check sorting
    if 'sortModel' in state and state['sortModel']:
        for sort in state['sortModel']:
            col = sort['colId']
            direction = 'descending' if sort['sort'] == 'desc' else 'ascending'
            explanations.append(f"Sorted by {col} ({direction})")

    # Check grouping
    if 'rowGroupColumns' in state and state['rowGroupColumns']:
        cols = ', '.join(state['rowGroupColumns'])
        explanations.append(f"Grouped by {cols}")

    if not explanations:
        return f"Applied query: {query}"

    return ' | '.join(explanations)

if __name__ == '__main__':
    print("🤖 PostNL AI API - Starting on http://localhost:8080")
    print(f"   Using Gemini 2.0 Flash (Free Tier)")
    print(f"   API Key: {GEMINI_API_KEY[:20]}...")
    app.run(host='0.0.0.0', port=8080, debug=True)
