import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="E-commerce Recommendation System",
    layout="wide"
)

# --------------------------------------------------
# Light styling
# --------------------------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 14px;
        border-radius: 14px;
    }

    .rec-card {
        background-color: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .rec-title {
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .rec-sub {
        font-size: 0.94rem;
        margin-bottom: 0.22rem;
        opacity: 0.92;
    }

    .helper-text {
        font-size: 0.97rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "RetailRocket")

# --------------------------------------------------
# Data loading
# --------------------------------------------------
@st.cache_data
def load_data():
    events = pd.read_csv(os.path.join(DATA_DIR, "events_filtered.csv"))
    item_popularity = pd.read_csv(os.path.join(DATA_DIR, "item_popularity.csv"))
    weighted_popularity = pd.read_csv(os.path.join(DATA_DIR, "weighted_popularity.csv"))
    item_similarity_df = pd.read_csv(
        os.path.join(DATA_DIR, "item_similarity_matrix.csv"),
        index_col=0
    )
    model_results = pd.read_csv(os.path.join(DATA_DIR, "model_comparison_results.csv"))

    metadata_path = os.path.join(DATA_DIR, "item_metadata.csv")
    if os.path.exists(metadata_path):
        item_metadata = pd.read_csv(metadata_path)
    else:
        item_metadata = pd.DataFrame()

    item_similarity_df.index = item_similarity_df.index.astype(int)
    item_similarity_df.columns = item_similarity_df.columns.astype(int)

    item_popularity["itemid"] = pd.to_numeric(item_popularity["itemid"], errors="coerce").astype("Int64")
    weighted_popularity["itemid"] = pd.to_numeric(weighted_popularity["itemid"], errors="coerce").astype("Int64")

    if not item_metadata.empty and "itemid" in item_metadata.columns:
        item_metadata["itemid"] = pd.to_numeric(item_metadata["itemid"], errors="coerce").astype("Int64")

    events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms", errors="coerce")
    events["event"] = events["event"].astype(str).str.strip().str.lower()

    return events, item_popularity, weighted_popularity, item_similarity_df, model_results, item_metadata


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def recommend_top_items(df, n=10):
    return df.head(n).reset_index(drop=True)


def recommend_similar_items(item_id, similarity_df, n=10):
    if item_id not in similarity_df.index:
        return pd.DataFrame({"message": [f"Item {item_id} not found in the similarity matrix."]})

    similar_scores = similarity_df[item_id].sort_values(ascending=False).iloc[1:n+1]
    return pd.DataFrame({
        "recommended_itemid": similar_scores.index,
        "similarity_score": similar_scores.values
    }).reset_index(drop=True)


def add_metadata_to_popularity(df, item_metadata):
    if df.empty or item_metadata.empty:
        return df
    return df.merge(item_metadata, on="itemid", how="left")


def add_metadata_to_recommendations(df, item_metadata):
    if df.empty or item_metadata.empty or "recommended_itemid" not in df.columns:
        return df
    merged = df.merge(item_metadata, left_on="recommended_itemid", right_on="itemid", how="left")
    if "itemid" in merged.columns:
        merged = merged.drop(columns=["itemid"])
    return merged


def itemid_to_str(series):
    return series.astype("Int64").astype(str)


def rename_popularity_table(df, score_col):
    rename_map = {
        "itemid_display": "Item ID",
        "interaction_count": "Total Interactions",
        "weighted_score": "Weighted Engagement Score",
        "category_label": "Category Group",
        "parent_category_label": "Broader Category Group",
        "property_count": "Known Attribute Count",
        "property_preview": "Key Attributes"
    }
    cols = [c for c in [
        "itemid_display",
        score_col,
        "category_label",
        "parent_category_label",
        "property_count",
        "property_preview"
    ] if c in df.columns]
    return df[cols].rename(columns=rename_map)


def rename_similarity_table(df):
    rename_map = {
        "recommended_itemid_display": "Recommended Item ID",
        "similarity_score": "Similarity Score",
        "category_label": "Category Group",
        "parent_category_label": "Broader Category Group",
        "property_count": "Known Attribute Count",
        "property_preview": "Key Attributes"
    }
    cols = [c for c in [
        "recommended_itemid_display",
        "similarity_score",
        "category_label",
        "parent_category_label",
        "property_count",
        "property_preview"
    ] if c in df.columns]
    return df[cols].rename(columns=rename_map)


def rename_popular_items_lookup_table(df):
    rename_map = {
        "itemid_display": "Item ID",
        "interaction_count": "Total Interactions",
        "category_label": "Category Group",
        "parent_category_label": "Broader Category Group",
        "property_count": "Known Attribute Count",
        "property_preview": "Key Attributes"
    }
    cols = [c for c in [
        "itemid_display",
        "interaction_count",
        "category_label",
        "parent_category_label",
        "property_count",
        "property_preview"
    ] if c in df.columns]
    return df[cols].rename(columns=rename_map)


def rename_model_results_table(df):
    rename_map = {
        "model": "Model",
        "hit_rate_at_10": "Hit Rate at 10",
        "recall_at_10": "Recall at 10"
    }
    return df.rename(columns=rename_map)


def render_featured_popularity_cards(df, score_col, reason_text, n_cards=4):
    if df.empty:
        st.info("No recommendations available.")
        return

    featured = df.head(n_cards)
    cols = st.columns(2)

    score_label = "Total Interactions" if score_col == "interaction_count" else "Weighted Engagement Score"

    for i, (_, row) in enumerate(featured.iterrows()):
        with cols[i % 2]:
            category = row["category_label"] if "category_label" in row and pd.notna(row["category_label"]) else "Unknown"
            parent = row["parent_category_label"] if "parent_category_label" in row and pd.notna(row["parent_category_label"]) else "Unknown"
            prop_count = int(row["property_count"]) if "property_count" in row and pd.notna(row["property_count"]) else 0
            prop_preview = row["property_preview"] if "property_preview" in row and pd.notna(row["property_preview"]) else "No attribute preview available"
            item_id_display = str(row["itemid"])

            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title">Item {item_id_display}</div>
                <div class="rec-sub"><b>{score_label}:</b> {row[score_col]:.2f}</div>
                <div class="rec-sub"><b>Category Group:</b> {category}</div>
                <div class="rec-sub"><b>Broader Category Group:</b> {parent}</div>
                <div class="rec-sub"><b>Known Attribute Count:</b> {prop_count}</div>
                <div class="rec-sub"><b>Key Attributes:</b> {prop_preview}</div>
                <div class="rec-sub"><b>Why it appears here:</b> {reason_text}</div>
            </div>
            """, unsafe_allow_html=True)


def render_featured_similarity_cards(df, selected_item, n_cards=4):
    if df.empty:
        st.info("No similar items found.")
        return

    featured = df.head(n_cards)
    cols = st.columns(2)

    for i, (_, row) in enumerate(featured.iterrows()):
        with cols[i % 2]:
            category = row["category_label"] if "category_label" in row and pd.notna(row["category_label"]) else "Unknown"
            parent = row["parent_category_label"] if "parent_category_label" in row and pd.notna(row["parent_category_label"]) else "Unknown"
            prop_count = int(row["property_count"]) if "property_count" in row and pd.notna(row["property_count"]) else 0
            prop_preview = row["property_preview"] if "property_preview" in row and pd.notna(row["property_preview"]) else "No attribute preview available"
            rec_item_display = str(row["recommended_itemid"])

            reason = f"Users who interacted with item {selected_item} also tended to interact with this item."
            if "category_label" in row and pd.notna(row["category_label"]):
                reason += f" It belongs to {category}."

            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title">Recommended Item {rec_item_display}</div>
                <div class="rec-sub"><b>Similarity Score:</b> {row['similarity_score']:.3f}</div>
                <div class="rec-sub"><b>Category Group:</b> {category}</div>
                <div class="rec-sub"><b>Broader Category Group:</b> {parent}</div>
                <div class="rec-sub"><b>Known Attribute Count:</b> {prop_count}</div>
                <div class="rec-sub"><b>Key Attributes:</b> {prop_preview}</div>
                <div class="rec-sub"><b>Why it was recommended:</b> {reason}</div>
            </div>
            """, unsafe_allow_html=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------
try:
    events, item_popularity, weighted_popularity, item_similarity_df, model_results, item_metadata = load_data()
except Exception as e:
    st.error("Could not load required project files.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Data Insights",
        "Popularity Recommender",
        "Similar Items Recommender",
        "Model Comparison",
        "Business Insights"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Project Summary")
st.sidebar.write(
    "Behavior-based e-commerce recommendation system using popularity baselines and item-based collaborative filtering."
)

# --------------------------------------------------
# Overview
# --------------------------------------------------
if page == "Overview":
    st.title("E-commerce Recommendation System")
    st.subheader("A retail recommendation prototype built from user interaction data")

    st.markdown(
        '<div class="helper-text">This dashboard compares two recommendation approaches: globally popular items and item-based similarity using user behavior patterns.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Filtered Events", f"{len(events):,}")
    col2.metric("Unique Users", f"{events['visitorid'].nunique():,}")
    col3.metric("Unique Items", f"{events['itemid'].nunique():,}")

    st.write(
        "This project explores how online retailers can improve product discovery using behavioral signals such as views, cart additions, and purchases."
    )

    if not model_results.empty and "hit_rate_at_10" in model_results.columns:
        try:
            pop_val = model_results.loc[
                model_results["model"] == "Popularity Baseline", "hit_rate_at_10"
            ].iloc[0]

            cf_val = model_results.loc[
                model_results["model"] == "Item-Based Collaborative Filtering", "hit_rate_at_10"
            ].iloc[0]

            if pop_val > 0:
                st.success(
                    f"Collaborative filtering improved Hit Rate at 10 from {pop_val*100:.2f}% to {cf_val*100:.2f}% "
                    f"(~{cf_val/pop_val:.1f}x improvement over the popularity baseline)."
                )

        except Exception:
            pass

# --------------------------------------------------
# Data Insights
# --------------------------------------------------
elif page == "Data Insights":
    st.title("Data Insights")
    st.markdown('<div class="helper-text">This page summarizes user behavior in the filtered interaction dataset. Use it to understand how people engage with items before looking at the recommendation outputs.</div>', unsafe_allow_html=True)

    event_counts = events["event"].astype(str).str.strip().value_counts().reset_index()
    event_counts.columns = ["Event Type", "Count"]

    color_map = {
        "view": "#60a5fa",
        "addtocart": "#f59e0b",
        "transaction": "#34d399"
    }

    fig = go.Figure()
    for _, row in event_counts.iterrows():
        fig.add_trace(go.Bar(
            x=[row["Event Type"]],
            y=[row["Count"]],
            name=row["Event Type"],
            marker_color=color_map.get(row["Event Type"], "#9ca3af"),
            text=[f"{row['Count']:,}"],
            textposition="outside"
        ))

    fig.update_layout(
        template="plotly_dark",
        height=420,
        showlegend=False,
        xaxis_title="Event Type",
        yaxis_title="Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        title="Distribution of User Actions"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("This chart shows how often each interaction type appears in the filtered dataset.")

    user_activity = events.groupby("visitorid").size()
    c1, c2, c3 = st.columns(3)
    c1.metric("Median User Interactions", f"{int(user_activity.median())}")
    c2.metric("Average User Interactions", f"{user_activity.mean():.2f}")
    c3.metric("Maximum User Interactions", f"{int(user_activity.max())}")

    top_items = item_popularity.head(10).copy()
    top_items["itemid_display"] = itemid_to_str(top_items["itemid"])

    fig2 = px.bar(
        top_items,
        x="interaction_count",
        y="itemid_display",
        orientation="h",
        color="interaction_count",
        color_continuous_scale="Blues",
        title="Top 10 Most Interacted-With Items"
    )
    fig2.update_layout(
        template="plotly_dark",
        height=450,
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="Total Interactions",
        yaxis_title="Item ID",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("This chart highlights the individual items with the highest overall interaction volume.")

    st.markdown("### Popular Items and Category Groups")
    st.markdown('<div class="helper-text">Use this lookup table to connect the most popular individual items with their category groups and available metadata.</div>', unsafe_allow_html=True)

    if not item_metadata.empty:
        popular_items_with_meta = top_items.merge(item_metadata, on="itemid", how="left")
        st.dataframe(
            rename_popular_items_lookup_table(popular_items_with_meta),
            use_container_width=True
        )
    else:
        st.info("Item metadata is not available.")

# --------------------------------------------------
# Popularity Recommender
# --------------------------------------------------
elif page == "Popularity Recommender":
    st.title("Popularity Recommender")
    st.markdown('<div class="helper-text">This view recommends items with the strongest overall engagement across all users. Use it to explore a simple non-personalized baseline.</div>', unsafe_allow_html=True)

    ranking_type = st.radio(
        "Select ranking method",
        ["Simple Popularity", "Weighted Popularity"],
        horizontal=True
    )
    n_recs = st.slider("Number of items to display", 4, 20, 8)

    if ranking_type == "Simple Popularity":
        recs = recommend_top_items(item_popularity, n=n_recs)
        score_col = "interaction_count"
        reason_text = "This item received strong overall engagement across users."
        chart_title = "Top Items by Total Interactions"
    else:
        recs = recommend_top_items(weighted_popularity, n=n_recs)
        score_col = "weighted_score"
        reason_text = "This item received stronger-intent interactions such as cart additions and purchases."
        chart_title = "Top Items by Weighted Engagement Score"

    recs = add_metadata_to_popularity(recs, item_metadata)
    recs["itemid_display"] = itemid_to_str(recs["itemid"])

    chart_df = recs.copy()

    fig = px.bar(
        chart_df,
        x=score_col,
        y="itemid_display",
        orientation="h",
        color=score_col,
        color_continuous_scale="Sunset",
        title=chart_title
    )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="Score",
        yaxis_title="Item ID",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Higher scores indicate stronger overall engagement across the full user base.")

    st.markdown("### Featured Results")
    render_featured_popularity_cards(recs, score_col, reason_text, n_cards=4)

    with st.expander("See full recommendation table"):
        st.dataframe(rename_popularity_table(recs, score_col), use_container_width=True)

# --------------------------------------------------
# Similar Items Recommender
# --------------------------------------------------
elif page == "Similar Items Recommender":
    st.title("Similar Items Recommender")
    st.markdown('<div class="helper-text">Select an item ID to find products that users tend to engage with in similar ways. This view demonstrates item-based collaborative filtering.</div>', unsafe_allow_html=True)

    available_items = sorted(item_similarity_df.index.tolist())
    selected_item = st.selectbox("Select an item ID", [str(x) for x in available_items])
    selected_item_int = int(selected_item)
    n_similar = st.slider("Number of similar items", 4, 16, 8)

    recs = recommend_similar_items(selected_item_int, item_similarity_df, n=n_similar)
    recs = add_metadata_to_recommendations(recs, item_metadata)

    if "recommended_itemid" in recs.columns:
        recs["recommended_itemid_display"] = itemid_to_str(recs["recommended_itemid"])

    left, right = st.columns([1, 2])

    with left:
        st.metric("Selected Item ID", selected_item)
        if not item_metadata.empty:
            selected_meta = item_metadata[item_metadata["itemid"] == selected_item_int]
            if not selected_meta.empty:
                if "category_label" in selected_meta.columns:
                    st.write(f"**Category Group:** {selected_meta['category_label'].iloc[0]}")
                if "parent_category_label" in selected_meta.columns:
                    st.write(f"**Broader Category Group:** {selected_meta['parent_category_label'].iloc[0]}")
                if "property_preview" in selected_meta.columns:
                    st.write(f"**Key Attributes:** {selected_meta['property_preview'].iloc[0]}")

    with right:
        st.write(
            f"Use this page by selecting an item ID in the dropdown above. The system then recommends items that users who engaged with item **{selected_item}** also tended to engage with."
        )

    if "recommended_itemid_display" in recs.columns:
        chart_df = recs.copy()

        fig = px.bar(
            chart_df,
            x="similarity_score",
            y="recommended_itemid_display",
            orientation="h",
            color="similarity_score",
            color_continuous_scale="Purp",
            title="Most Similar Items by Similarity Score"
        )
        fig.update_layout(
            template="plotly_dark",
            height=450,
            yaxis=dict(categoryorder="total ascending"),
            xaxis_title="Similarity Score",
            yaxis_title="Recommended Item ID",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Higher similarity scores indicate stronger behavioral similarity between items.")

    if "category_label" in recs.columns:
        category_mix = recs["category_label"].value_counts().head(8).reset_index()
        category_mix.columns = ["Category Group", "Count"]

        fig2 = px.bar(
            category_mix,
            x="Count",
            y="Category Group",
            orientation="h",
            color="Count",
            color_continuous_scale="Magenta",
            title="Category Distribution of Recommended Items"
        )
        fig2.update_layout(
            template="plotly_dark",
            height=420,
            yaxis=dict(categoryorder="total ascending"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.caption("This chart shows whether recommendations stay concentrated in a small set of categories or spread across several groups.")

    st.markdown("### Featured Results")
    render_featured_similarity_cards(recs, selected_item, n_cards=4)

    with st.expander("See full recommendation table"):
        st.dataframe(rename_similarity_table(recs), use_container_width=True)

# --------------------------------------------------
# Model Comparison
# --------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown('<div class="helper-text">This page compares recommendation quality across the baseline and collaborative filtering approaches using offline evaluation metrics.</div>', unsafe_allow_html=True)

    comparison_df = model_results.copy()
    metric_cols = [col for col in comparison_df.columns if col != "model"]
    for col in metric_cols:
        comparison_df[col] = pd.to_numeric(comparison_df[col], errors="coerce")

    st.dataframe(rename_model_results_table(comparison_df), use_container_width=True)

    if "hit_rate_at_10" in comparison_df.columns:
        chart_df = comparison_df[["model", "hit_rate_at_10"]].copy()
        chart_df["hit_rate_pct"] = chart_df["hit_rate_at_10"] * 100

        c1, c2 = st.columns(2)
        pop_row = chart_df[chart_df["model"] == "Popularity Baseline"]
        cf_row = chart_df[chart_df["model"] == "Item-Based Collaborative Filtering"]

        if not pop_row.empty:
            c1.metric("Popularity Baseline", f"{pop_row['hit_rate_pct'].iloc[0]:.2f}%")
        if not cf_row.empty:
            c2.metric("Item-Based Collaborative Filtering", f"{cf_row['hit_rate_pct'].iloc[0]:.2f}%")

        x_vals = chart_df["hit_rate_pct"].tolist()
        y_vals = chart_df["model"].tolist()
        labels = [f"{x:.2f}%" for x in x_vals]
        max_val = max(x_vals) if len(x_vals) > 0 else 1
        upper_bound = max(max_val * 1.25, 3.0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+text",
            text=labels,
            textposition="middle right",
            marker=dict(size=18, color=["#f59e0b", "#60a5fa"]),
            showlegend=False
        ))
        fig.update_layout(
            title="Hit Rate at 10 by Model",
            template="plotly_dark",
            height=420,
            xaxis_title="Hit Rate at 10 (%)",
            yaxis_title="Model",
            xaxis=dict(range=[0, upper_bound]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption("In this project setup, Recall at 10 is effectively the same as Hit Rate at 10 because one item is held out per user.")

# --------------------------------------------------
# Business Insights
# --------------------------------------------------
elif page == "Business Insights":
    st.title("Business Insights")

    st.markdown(
        '<div class="helper-text">This page summarizes what the recommendation results mean from a product and business perspective.</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This project shows how recommendation systems can move from generic product ranking toward more behavior-driven personalization in an e-commerce setting."
    )

    st.write("- Popularity-based methods are useful for trending and cold-start situations.")
    st.write("- Collaborative filtering captures relationships between products based on real user behavior.")
    st.write("- Similarity-based recommendations surface related products based on real user behavior, enabling more targeted and relevant product discovery.")
    st.write("- Metadata enrichment improves interpretability even when product names are unavailable.")
    st.write("- A strong next step would be a hybrid model combining similarity with metadata.")

    # -----------------------------------------
    # Performance Summary 
    # -----------------------------------------
    if not model_results.empty and "hit_rate_at_10" in model_results.columns:
        biz_df = model_results.copy()
        biz_df["hit_rate_at_10"] = pd.to_numeric(biz_df["hit_rate_at_10"], errors="coerce")
        biz_df["hit_rate_pct"] = biz_df["hit_rate_at_10"] * 100

        st.markdown("### Performance Summary")

        col1, col2 = st.columns(2)

        pop_row = biz_df[biz_df["model"] == "Popularity Baseline"]
        cf_row = biz_df[biz_df["model"] == "Item-Based Collaborative Filtering"]

        if not pop_row.empty:
            col1.metric(
                "Popularity Baseline",
                f"{pop_row['hit_rate_pct'].iloc[0]:.2f}%"
            )

        if not cf_row.empty:
            col2.metric(
                "Item-Based Collaborative Filtering",
                f"{cf_row['hit_rate_pct'].iloc[0]:.2f}%"
            )

        # Improvement explanation
        if not pop_row.empty and not cf_row.empty:
            pop_val = pop_row["hit_rate_at_10"].iloc[0]
            cf_val = cf_row["hit_rate_at_10"].iloc[0]

            if pop_val > 0:
                improvement = cf_val / pop_val
                st.write(
                    f"In this evaluation setup, collaborative filtering outperformed the popularity baseline by about {improvement:.1f}x on Hit Rate at 10."
                )

        
