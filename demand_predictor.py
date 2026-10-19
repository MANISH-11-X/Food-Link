
from datetime import datetime, timedelta

import numpy as np
from flask import render_template, session, redirect, flash
from sklearn.linear_model import LinearRegression

HISTORY_DAYS = 30      # how many past days we learn from
FORECAST_DAYS = 7      # how many future days we predict
MIN_REAL_DAYS = 7      # need at least this many days with data, else use demo data


# ---------------------------------------------------------------
# STEP 1: Get past demand (requests per day) from MongoDB
# ---------------------------------------------------------------
def get_daily_history(requests_collection):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start = today - timedelta(days=HISTORY_DAYS - 1)

    # a dictionary like {date: 0} for each of the last 30 days
    counts = {start + timedelta(days=i): 0 for i in range(HISTORY_DAYS)}

    for req in requests_collection.find({"requested_at": {"$gte": start}}):
        day = req["requested_at"].replace(hour=0, minute=0, second=0, microsecond=0)
        if day in counts:
            counts[day] += 1

    dates = list(counts.keys())
    values = list(counts.values())
    return dates, values


# ---------------------------------------------------------------
# STEP 2: Demo data (used when the database has too little data)
# ---------------------------------------------------------------
def make_demo_history():
    rng = np.random.default_rng(42)          # fixed seed = same numbers every time
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = [today - timedelta(days=HISTORY_DAYS - 1 - i) for i in range(HISTORY_DAYS)]

    values = []
    for i, d in enumerate(dates):
        demand = 5 + 0.15 * i                # slow upward trend
        if d.weekday() >= 5:                 # weekends have more demand
            demand += 4
        demand += rng.normal(0, 1)           # random noise
        values.append(max(0, round(demand)))
    return dates, values


# ---------------------------------------------------------------
# STEP 3: Turn a date into numbers the model can understand
# ---------------------------------------------------------------
def make_features(day_number, date):
    is_weekend = 1 if date.weekday() >= 5 else 0
    return [day_number, is_weekend]


# ---------------------------------------------------------------
# STEP 4: Train the model and predict the next 7 days
# ---------------------------------------------------------------
def forecast_demand(dates, values):
    # X = inputs (day number, is_weekend)   y = answer (requests that day)
    X = np.array([make_features(i, d) for i, d in enumerate(dates)])
    y = np.array(values)

    model = LinearRegression()
    model.fit(X, y)                                   # <- the "learning" step

    last_index = len(dates) - 1
    future_dates = [dates[-1] + timedelta(days=i + 1) for i in range(FORECAST_DAYS)]
    X_future = np.array(
        [make_features(last_index + i + 1, d) for i, d in enumerate(future_dates)]
    )

    predictions = model.predict(X_future)             # <- the "predicting" step
    predictions = [round(max(0, float(p)), 1) for p in predictions]   # no negatives

    return {
        "future_dates": future_dates,
        "predictions": predictions,
        "trend_per_day": float(model.coef_[0]),       # + = demand growing
        "weekend_boost": float(model.coef_[1]),       # extra demand on weekends
        "accuracy": round(max(0.0, model.score(X, y)) * 100, 1),   # R2 as %
    }


# ---------------------------------------------------------------
# STEP 5: Which food types are requested the most?
# ---------------------------------------------------------------
def demand_by_food_type(db):
    counts = {}
    for req in db["pickup_requests"].find():
        food = db["surplus"].find_one({"_id": req.get("food_id")})
        food_type = food.get("food_type", "Other") if food else "Other"
        counts[food_type] = counts.get(food_type, 0) + 1

    if not counts:   # demo fallback
        counts = {"Cooked": 18, "Raw": 9, "Packaged": 12, "Bakery": 6}
    return counts


# ---------------------------------------------------------------
# STEP 6: Build all data for the page
# ---------------------------------------------------------------
def build_demand_report(db):
    dates, values = get_daily_history(db["pickup_requests"])

    real_days = sum(1 for v in values if v > 0)
    using_demo = real_days < MIN_REAL_DAYS
    if using_demo:
        dates, values = make_demo_history()

    result = forecast_demand(dates, values)
    predictions = result["predictions"]

    peak_index = int(np.argmax(predictions))
    trend = result["trend_per_day"]

    return {
        "using_demo": using_demo,
        "history_labels": [d.strftime("%d %b") for d in dates],
        "history_values": values,
        "forecast_labels": [d.strftime("%d %b") for d in result["future_dates"]],
        "forecast_values": predictions,
        "food_types": demand_by_food_type(db),
        "total_next_week": round(sum(predictions)),
        "peak_day": result["future_dates"][peak_index].strftime("%A, %d %b"),
        "peak_value": predictions[peak_index],
        "trend_text": "Growing" if trend > 0.05 else "Falling" if trend < -0.05 else "Stable",
        "trend_per_day": round(trend, 2),
        "weekend_boost": round(result["weekend_boost"], 1),
        "accuracy": result["accuracy"],
    }


# ---------------------------------------------------------------
# STEP 7: Connect the page to Flask
# ---------------------------------------------------------------

def register_demand_routes(app, db):

    @app.route("/food-demand")
    def food_demand():
        # only logged-in donors or NGOs may view
        if "donor_id" not in session and "ngo_id" not in session:
            flash("Please login first.")
            return redirect("/")

        report = build_demand_report(db)
        return render_template("food_demand.html", report=report)
