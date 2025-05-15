
import math
EARTH_RADIUS_MILES = 3958.8

def haversine(lon1, lat1, lon2, lat2):
    """Great‑circle distance in miles."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return EARTH_RADIUS_MILES * 2 * math.asin(math.sqrt(a))

def warehousing_cost(demand_lbs, sqft_per_lb, cost_per_sqft, fixed_cost):
    return fixed_cost + demand_lbs * sqft_per_lb * cost_per_sqft

# ───────────────────────── ORS drive time helper ─────────────────────────
def get_drive_time_matrix(origins, destinations, api_key):
    """Returns a matrix of drive times in **seconds** using OpenRouteService.

    If the request fails (no key / quota), returns None so the caller can
    fall back to straight‑line minutes.
    """
    if not api_key:
        return None
    try:
        import openrouteservice
        client = openrouteservice.Client(key=api_key)
        matrix = client.distance_matrix(
            locations=origins + destinations,
            profile="driving-car",
            metrics=["duration"],
            sources=list(range(len(origins))),
            destinations=list(range(len(origins), len(origins) + len(destinations))),
        )
        return matrix["durations"]
    except Exception as e:
        print("ORS error", e)
        return None
