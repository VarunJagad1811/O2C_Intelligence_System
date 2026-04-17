import os
import streamlit as st
import concurrent.futures
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# ✅ FINAL API SETUP (ONLY THIS)
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ==========================================
# === 1. DEFINE AGENTIC TOOLS (AGENCY & RAG)
# ==========================================

@tool
def query_live_shipping_rates(weight_kg: float, current_mode: str) -> str:
    """Always use this tool to check live logistics costs before making a financial decision."""
    # Simulates an ERP pricing engine
    if current_mode.lower() == 'air' and weight_kg > 10:
        return "ERP ALERT: Air freight for items over 10kg is surging today. Switching to Ground will save $145.00 and automatically bypass TSA aviation holds."
    return "ERP DATA: Current shipping mode is within normal cost parameters. No urgent cost-savings found."

@tool
def search_policy_database(search_query: str) -> str:
    """Always use this tool to search the enterprise compliance database for shipping regulations before advising."""
    query = search_query.lower()
    
    # 1. Large Electronics (E-Waste & Heavy Hazmat)
    if "large electronic" in query or ("electronic" in query and "large" in query):
        return "VECTOR DB [WEEE-Dir-2012]: Large electronics require WEEE recycling compliance documentation and UN3480 large lithium battery freight declarations."
        
    # 2. Small Electronics (Theft & Standard Battery)
    elif "small electronic" in query or "electronic" in query or "battery" in query:
        return "VECTOR DB [IATA-UN3481]: Small electronics require UN3481 lithium battery handling labels and high-security poly-bagging to prevent transit theft."
        
    # 3. Apparel (Textile Compliance)
    elif "apparel" in query or "clothing" in query or "textile" in query:
        return "VECTOR DB [CPSC-Textile-16]: Apparel shipments must include a Certificate of Origin and verify Class 1 flammability standards compliance."
        
    # 4. Home Goods (Fragile & Wood Packing)
    elif "home good" in query or "furniture" in query or "decor" in query:
        return "VECTOR DB [ISPM-15-Wood]: Home goods containing wood components require ISPM-15 heat-treatment certification to prevent port quarantine holds."
        
    # 5. High-Value Orders (C-TPAT / Insurance)
    elif "value" in query or "expensive" in query:
        return "VECTOR DB [C-TPAT-HighValue]: Orders exceeding $2,000 USD require 'Signature Required' delivery, GPS tracking, and supplemental transit insurance."
        
    # 6. Aviation Security & Weight (IATA)
    elif "air" in query or "weight" in query or "heavy" in query:
        return "VECTOR DB [IATA-Sec-4]: Air freight exceeding 20kg is flagged for mandatory secondary X-ray screening and dimensional weight auditing."
        
    # 7. Fallback (Standard Operations)
    return "VECTOR DB [SOP-01]: Standard domestic shipping protocols apply. Proceed with standard dispatch."

# ==========================================
# === 2. MASTER AGENT EXECUTION ENGINE ===
# ==========================================

def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def _run_agent_with_tools(llm, system_prompt, user_content, tools_list, final_formatting_prompt=""):
    """
    A mature LangChain executor that handles tool binding, execution, and history automatically.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    # If no tools are assigned, just get standard LLM text
    if not tools_list:
        final_answer = llm.invoke(messages).content.replace("```html", "").replace("```", "").strip()
        return final_answer, ""

    # Bind tools and ask LLM to think
    llm_with_tools = llm.bind_tools(tools_list)
    response = llm_with_tools.invoke(messages)
    
    ui_logs = ""
    
    # If the AI decides to use a tool, execute it autonomously
    if response.tool_calls:
        messages.append(response) # Save the AI's tool request
        
        for tool_call in response.tool_calls:
            t_name = tool_call['name']
            t_args = tool_call['args']
            
            # Route to the actual Python function
            if t_name == 'search_policy_database':
                t_result = search_policy_database.invoke(t_args)
                ui_color = "#3b82f6" # Blue for Compliance DB (RAG)
                ui_title = "🔍 RAG DATABASE QUERY:"
            elif t_name == 'query_live_shipping_rates':
                t_result = query_live_shipping_rates.invoke(t_args)
                ui_color = "#a855f7" # Purple for ERP Call
                ui_title = "⚙️ AUTONOMOUS ERP QUERY:"
            else:
                t_result = "Tool not found."
                ui_color = "#94a3b8"
                ui_title = "❓ UNKNOWN TOOL:"

            # ZERO-INDENTATION UI Log to prevent Markdown bugs
            ui_logs += f"""
<div style='margin-bottom: 10px; font-size: 0.85rem; border-left: 2px solid {ui_color}; padding-left: 10px; background: rgba(255,255,255, 0.02); padding-top: 8px; padding-bottom: 8px;'>
<span style='color: {ui_color}; font-weight: bold;'>{ui_title}</span> <span style='color: #94a3b8;'><i>{t_name}({t_args})</i></span><br>
<span style='color: #cbd5e1;'>Result ➔ {t_result}</span>
</div>
"""
            # Inject the external data back into the LLM
            messages.append(ToolMessage(content=t_result, tool_call_id=tool_call['id']))
        
        # Ask LLM for the final answer now that it has the tool data
        messages.append(HumanMessage(content=f"Using the data retrieved, provide your final response. DO NOT use markdown lists or bullet points. {final_formatting_prompt}"))
        final_answer = llm.invoke(messages).content.replace("```html", "").replace("```", "").strip()
        return final_answer, ui_logs
    else:
        # Fallback if LLM decides no tools are needed
        messages.append(HumanMessage(content=f"DO NOT use markdown lists or bullet points. {final_formatting_prompt}"))
        final_answer = llm.invoke(messages).content.replace("```html", "").replace("```", "").strip()
        return final_answer, ""

# ==========================================
# === 3. ORCHESTRATORS (TAB 3 & TAB 4) ===
# ==========================================

# ... [Keep your tools and _run_agent_with_tools functions exactly as they are] ...

def generate_risk_narrative(risk_score, metadata_dict, top_factors):
    try:
        llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model_name="llama-3.1-8b-instant", temperature=0.1)

        drivers_text = ", ".join([f"{f['feature']} (+{f['val']*100:.1f}%)" for f in top_factors]) if top_factors else "None"
        o_val = metadata_dict.get('order_value', 0)
        o_weight = float(metadata_dict.get('package_weight_kg', 0))
        o_mode = str(metadata_dict.get('shipping_mode', 'Unknown'))
        o_intl = 'Yes' if metadata_dict.get('is_international') == 1 else 'No'
        
        ctx = f"ORDER: ${o_val:.2f}, Cat: {metadata_dict.get('product_type')}, Mode: {o_mode}, Weight: {o_weight}kg, Intl: {o_intl}. RISK: {risk_score:.1%}. DRIVERS: {drivers_text}"

        comp_sys = "You are a Compliance Officer. You MUST use your search_policy_database tool to check enterprise regulations before giving advice."
        comp_prompt = f"Analyze this order and state the primary compliance risk in one short sentence, citing the database. Context: {ctx}"

        log_sys = "You are a Logistics Manager. Focus on speed and avoiding manual bottlenecks."
        log_prompt = f"Identify the #1 logistical priority for this order in one short sentence. Context: {ctx}"

        # 1. PARALLEL ADVISORY NODES
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_comp = executor.submit(_run_agent_with_tools, llm, comp_sys, comp_prompt, [search_policy_database], "Keep it to one sentence.")
            future_log = executor.submit(_run_agent_with_tools, llm, log_sys, log_prompt, None, "Keep it to one sentence.")
            
            comp_advice, comp_logs = future_comp.result()
            log_advice, log_logs = future_log.result()

        # 2. FINANCE DIRECTOR (First Draft)
        fin_sys = "You are the Finance Director. You MUST use your query_live_shipping_rates tool to check costs before deciding."
        fin_prompt = f"Compliance: '{comp_advice}'\nLogistics: '{log_advice}'\nWeight: {o_weight}kg. Mode: {o_mode}.\nDetermine the plan."
        fin_format = "Use EXACTLY this raw HTML: <ul><li style='margin-bottom: 8px;'><b>The Situation:</b> [Summary]</li><li><b>The Solution:</b> [Action step]</li></ul>"
        
        draft_verdict, fin_logs = _run_agent_with_tools(llm, fin_sys, fin_prompt, [query_live_shipping_rates], fin_format)

        # 3. THE REFLECTION NODE (Cyclic Agentic Workflow)
        audit_sys = "You are a strict Risk Auditor. Review the Finance Director's plan. If it is safe and logical, reply exactly with 'APPROVED'. If it creates a new delay or ignores compliance, reply with 'REJECTED: [Reason]'."
        audit_prompt = f"Draft Plan:\n{draft_verdict}\n\nDoes this effectively balance cost, speed, and compliance? Keep it under 20 words."
        
        audit_response = llm.invoke([SystemMessage(content=audit_sys), HumanMessage(content=audit_prompt)]).content.strip()

        reflection_logs = ""
        final_verdict = draft_verdict

        if "REJECTED" in audit_response.upper():
            # If rejected, loop back to Finance for a rewrite!
            reflection_logs = f"""
<div style='margin-bottom: 15px; font-size: 0.85rem; border-left: 2px solid #ef4444; padding-left: 10px; background: rgba(239, 68, 68, 0.05); padding-top: 8px; padding-bottom: 8px;'>
<span style='color: #ef4444; font-weight: bold;'>🛑 AUDITOR REJECTION:</span> <span style='color: #cbd5e1;'>"{audit_response}"</span><br>
<span style='color: #94a3b8; font-style: italic;'>Forcing Finance Director to revise the plan...</span>
</div>
"""
            rewrite_sys = "You are the Finance Director. Your previous plan was rejected. You must revise it."
            rewrite_prompt = f"Original Plan:\n{draft_verdict}\n\nRejection Reason:\n{audit_response}\n\nProvide a NEW plan that fixes this flaw. {fin_format}"
            final_verdict = llm.invoke([SystemMessage(content=rewrite_sys), HumanMessage(content=rewrite_prompt)]).content.replace("```html", "").replace("```", "").strip()
        else:
            # Approved on first try
            reflection_logs = f"""
<div style='margin-bottom: 15px; font-size: 0.85rem; border-left: 2px solid #10b981; padding-left: 10px; background: rgba(16, 185, 129, 0.05); padding-top: 8px; padding-bottom: 8px;'>
<span style='color: #10b981; font-weight: bold;'>✅ AUDITOR APPROVED:</span> <span style='color: #cbd5e1;'>Plan passes all SLA and compliance checks on the first iteration.</span>
</div>
"""

        color = "#f87171" if risk_score > 0.5 else "#4ade80"
        
        return f"""
<div style='background: rgba(255, 255, 255, 0.03); padding: 20px; border-radius: 10px; border-left: 4px solid {color}; font-size: 0.95rem; line-height: 1.6;'>
<strong style='color: {color}; font-size: 1.1rem; margin-bottom: 15px; display: block;'>🧠 LangChain Autonomous AI Workflows</strong>
{comp_logs}
<div style='margin-bottom: 10px; font-size: 0.85rem; border-left: 2px solid #eab308; padding-left: 10px;'>
<span style='color: #eab308; font-weight: bold;'>🛡️ COMPLIANCE:</span> <span style='color: #cbd5e1;'>"{comp_advice}"</span>
</div>
<div style='margin-bottom: 15px; font-size: 0.85rem; border-left: 2px solid #38bdf8; padding-left: 10px;'>
<span style='color: #38bdf8; font-weight: bold;'>📦 LOGISTICS:</span> <span style='color: #cbd5e1;'>"{log_advice}"</span>
</div>
{fin_logs}
{reflection_logs}
<div style='background: rgba(255,255,255,0.03); padding: 10px; border-radius: 5px;'>
<span style='font-weight: bold; color: {color};'>💼 FINAL FINANCE RULING:</span><br>
<span style='color: #f1f5f9;'>{final_verdict}</span>
</div>
</div>
"""
    except Exception as e:
        return f"⚠️ **LangChain API Error:** {str(e)}"

# Update generate_detailed_business_report with the exact same reflection logic
def generate_detailed_business_report(case_id, risk_score, metadata_dict, top_factors):
    try:
        llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model_name="llama-3.1-8b-instant", temperature=0.1)

        drivers_text = ", ".join([f"{f['feature']}" for f in top_factors]) if top_factors else "None"
        o_val = metadata_dict.get('order_value', 0)
        o_weight = float(metadata_dict.get('package_weight_kg', 0))
        o_mode = str(metadata_dict.get('shipping_mode', 'Unknown'))
        o_intl = 'Yes' if metadata_dict.get('is_international') == 1 else 'No'
        
        ctx = f"CASE ID: {case_id}. Value ${o_val:.2f}, Mode: {o_mode}, Weight: {o_weight}kg, Intl: {o_intl}. RISK: {risk_score:.1%}. DRIVERS: {drivers_text}"

        comp_sys = "You are a Compliance Officer. You MUST use your search_policy_database tool to check regulations."
        comp_prompt = f"Analyze this order and state the compliance risk in one sentence. Context: {ctx}"

        log_sys = "You are a Logistics Manager. Focus on speed and delays."
        log_prompt = f"Identify the operational risk in one sentence. Context: {ctx}"

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_comp = executor.submit(_run_agent_with_tools, llm, comp_sys, comp_prompt, [search_policy_database], "Keep it to one sentence.")
            future_log = executor.submit(_run_agent_with_tools, llm, log_sys, log_prompt, None, "Keep it to one sentence.")
            
            comp_advice, comp_logs = future_comp.result()
            log_advice, log_logs = future_log.result()

        fin_sys = "You are the Finance Director writing an executive action plan. You MUST use tools to check financial data."
        fin_prompt = f"Compliance: '{comp_advice}'\nLogistics: '{log_advice}'\nWeight: {o_weight}kg. Mode: {o_mode}.\nDetermine the plan."
        fin_format = "Use EXACTLY this raw HTML: <ul><li style='margin-bottom: 8px;'><b>Executive Summary:</b> [1 sentence]</li><li style='margin-bottom: 8px;'><b>Recommended Action:</b> [Action step]</li><li><b>Expected Benefit:</b> [Savings]</li></ul>"
        
        draft_verdict, fin_logs = _run_agent_with_tools(llm, fin_sys, fin_prompt, [query_live_shipping_rates], fin_format)

        # Reflection Node
        audit_sys = "You are a strict Risk Auditor. Review the plan. Reply 'APPROVED' if good. If it creates delays, reply 'REJECTED: [Reason]'."
        audit_prompt = f"Draft Plan:\n{draft_verdict}\n\nDoes this balance cost, speed, and compliance?"
        
        audit_response = llm.invoke([SystemMessage(content=audit_sys), HumanMessage(content=audit_prompt)]).content.strip()

        reflection_logs = ""
        final_verdict = draft_verdict

        if "REJECTED" in audit_response.upper():
            reflection_logs = f"""
<div style='margin-bottom: 15px; font-size: 0.95rem; border-left: 2px solid #ef4444; padding-left: 10px; background: rgba(239, 68, 68, 0.05); padding-top: 8px; padding-bottom: 8px;'>
<span style='color: #ef4444; font-weight: bold;'>🛑 AUDITOR REJECTION:</span> <span style='color: #cbd5e1;'>"{audit_response}"</span><br>
<span style='color: #94a3b8; font-style: italic;'>Forcing Finance Director to revise the plan...</span>
</div>
"""
            rewrite_sys = "You are the Finance Director. Your previous plan was rejected. You must revise it."
            rewrite_prompt = f"Original Plan:\n{draft_verdict}\n\nRejection Reason:\n{audit_response}\n\nProvide a NEW plan. {fin_format}"
            final_verdict = llm.invoke([SystemMessage(content=rewrite_sys), HumanMessage(content=rewrite_prompt)]).content.replace("```html", "").replace("```", "").strip()
        else:
            reflection_logs = f"""
<div style='margin-bottom: 15px; font-size: 0.95rem; border-left: 2px solid #10b981; padding-left: 10px; background: rgba(16, 185, 129, 0.05); padding-top: 8px; padding-bottom: 8px;'>
<span style='color: #10b981; font-weight: bold;'>✅ AUDITOR APPROVED:</span> <span style='color: #cbd5e1;'>Plan passes all SLA and compliance checks on the first iteration.</span>
</div>
"""

        return f"""
<div class='report-box'>
<strong style='color: #f8fafc; font-size: 1.2rem; margin-bottom: 15px; display: block;'>📑 Multi-Agent Executive Report for {case_id}</strong>
{comp_logs}
<div style='margin-bottom: 12px; font-size: 0.95rem; border-left: 2px solid #eab308; padding-left: 10px;'>
<span style='color: #eab308; font-weight: bold;'>🛡️ COMPLIANCE AUDIT:</span> <span style='color: #cbd5e1;'>"{comp_advice}"</span>
</div>
<div style='margin-bottom: 18px; font-size: 0.95rem; border-left: 2px solid #38bdf8; padding-left: 10px;'>
<span style='color: #38bdf8; font-weight: bold;'>📦 LOGISTICS AUDIT:</span> <span style='color: #cbd5e1;'>"{log_advice}"</span>
</div>
{fin_logs}
{reflection_logs}
<div style='background: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px;'>
<span style='font-weight: bold; color: #4ade80;'>💼 FINAL FINANCE DIRECTOR'S ACTION PLAN:</span><br><br>
<span style='color: #f1f5f9; line-height: 1.6;'>{final_verdict}</span>
</div>
</div>
"""
    except Exception as e:
        return f"⚠️ **LangChain API Error:** {str(e)}"

def run_autonomous_agent(risk_score, case_inputs, unique_tab_id):
    """
    Maintains the existing deterministic rules for UI intervention rendering.
    """
    with st.container(border=True):
        st.markdown("#### ⚙️ Recommended Actions")
        
        val = case_inputs['order_value'].iloc[0]
        weight = case_inputs['package_weight_kg'].iloc[0]
        is_intl = case_inputs['is_international'].iloc[0]
        is_elec = case_inputs['is_large_electronic'].iloc[0]
        staff = case_inputs['staff_training_level'].iloc[0]
        vendor_score = case_inputs['vendor_reliability_score'].iloc[0]

        if risk_score > 0.50:
            st.markdown("<span style='color:#f87171; font-weight:bold;'>🔴 High Risk: Fix these issues to pass manual review quickly</span>", unsafe_allow_html=True)
            
            actions = []
            if staff == 0:
                actions.append("**Assign to Senior Staff:** Move this complex order to an experienced worker to ensure compliance documents are filled out perfectly.")
            if is_elec == 1:
                actions.append("**Hazmat / Lithium Battery Check:** Electronics often trigger aviation security reviews. Ensure battery declaration forms are pre-attached.")
            if is_intl == 1:
                actions.append("**Pre-clear Customs:** Get the export paperwork and commercial invoices ready now so it doesn't get stuck at the border.")
            if weight > 20:
                actions.append("**Aviation Weight Limit:** Heavy items trigger strict aviation dimensional checks. Split into smaller parcels or switch to Ground/Sea freight to bypass the hold.")
            if vendor_score < 80:
                actions.append("**Quality Assurance:** This vendor has a history of paperwork errors. Perform a secondary manual check before handing it to the carrier.")
            
            if len(actions) == 0:
                actions.append("**Proactive Communication:** The order is held up in review. Alert the customer now so they aren't surprised by a delayed delivery.")

            for act in actions:
                st.error(act, icon="🚨")

        else:
            st.markdown("<span style='color:#4ade80; font-weight:bold;'>🟢 Low Risk: Opportunities to make more money</span>", unsafe_allow_html=True)
            
            actions = []
            if val > 2000:
                actions.append("**Follow-up Email:** Send a polite thank-you email after delivery to keep this high-value customer happy.")
            if is_elec == 1:
                actions.append("**Offer Accessories:** Since they bought electronics, send them a discount code for related accessories.")
            if weight < 5 and is_intl == 0:
                actions.append("**Use Cheaper Packaging:** Pack this small item in a poly-mailer bag instead of a box to save on dimensional shipping costs.")
            if vendor_score >= 90:
                actions.append("**Skip Extra Checks:** This trusted vendor rarely makes mistakes; send it straight to shipping to save warehouse processing time.")
            if is_intl == 1:
                actions.append("**Wait and Combine:** Hold this for a few hours to ship it together with other international orders to save money on freight.")

            if len(actions) == 0:
                actions.append("**Find Cheaper Shipping:** The system can automatically check for a cheaper delivery service that still arrives on time.")

            for act in actions:
                st.success(act, icon="💡")
