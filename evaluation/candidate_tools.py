candidate_tools = [
  {
    "function_name": "AMAZON_search_products",
    "default_arguments": {},
    "function_comment": "Search Amazon products via Rainforest API and return a concise plain-text list.\nArgs:\n  keywords (str): Product search keywords.\n  n (int): Number of results to include (1–10).\n  page (int): Result page index starting from 1.\n  amazon_domain (str): Amazon domain such as \"amazon.com\".\nReturns:\n  result (str): Plain-text list of up to n products or an error message."
  },
  {
    "function_name": "CAR_PRICE_get_car_brands",
    "default_arguments": {},
    "function_comment": "Get all available car brands from FIPE via the fipe.online API.\nArgs:\n  (none)\nReturns:\n  result (str): JSON string containing the list of brand identifiers and names."
  },
  {
    "function_name": "FOOD_NUTRITION_MCP_get_nutrition",
    "default_arguments": {},
    "function_comment": "Get nutritional information for a described food item using Edamam APIs.\nArgs:\n  query (Optional[str]): Food name with quantity description.\n  food (Optional[str]): Alias for query when provided instead.\nReturns:\n  result (str): Summary of key nutrients and units or an error message."
  },
  {
    "function_name": "GOOGLE_MAPS_geocode",
    "default_arguments": {},
    "function_comment": "Forward geocode a human-readable address using Google Geocoding API.\nArgs:\n  address (str): Free-text address to geocode.\n  region (Optional[str]): Region bias such as \"us\".\n  language (Optional[str]): Response language code such as \"en\".\nReturns:\n  data (Dict[str, Any]): JSON with endpoint, request parameters, and API response."
  },
  {
    "function_name": "GOOGLE_MAPS_places_text_search",
    "default_arguments": {},
    "function_comment": "Search for places using a free-text query via Google Places Text Search.\nArgs:\n  query (str): Text query describing the place.\n  languageCode (Optional[str]): Response language code.\n  regionCode (Optional[str]): Region bias code.\n  locationBias (Optional[Dict[str, Any]]): Optional circle bias for location preference.\n  maxResultCount (int): Maximum number of results to keep client-side.\n  fields (Optional[str]): Unused field mask parameter for signature parity.\nReturns:\n  data (Dict[str, Any]): JSON with endpoint, request parameters, and trimmed Places response."
  },
  {
    "function_name": "HEALTHCARE_MCP_fda_drug_lookup",
    "default_arguments": {
      "search_type": "general"
    },
    "function_comment": "Look up FDA drug information by name.\nArgs:\n  drug_name (str): Drug name to search.\n  search_type (str): Information type to retrieve; one of \"general\" (default), \"label\", or \"adverse_events\".\nReturns:\n  result (object): JSON object with status, drug_name, total_results, and matching FDA drug records."
  },
  {
    "function_name": "HEALTHCARE_MCP_health_topics",
    "default_arguments": {
      "language": "en"
    },
    "function_comment": "Get evidence-based health topic summaries from Health.gov.\nArgs:\n  topic (str): Health topic search term.\n  language (str): Result language code; \"en\" (default) or \"es\".\nReturns:\n  result (object): JSON object with status, search_term, language, total_results, and matching topic entries."
  },
  {
    "function_name": "HEALTHCARE_MCP_lookup_icd_code",
    "default_arguments": {
      "max_results": 10
    },
    "function_comment": "Look up ICD-10 diagnosis codes and descriptions.\nArgs:\n  code (Optional[str]): ICD-10 code to look up; optional if description is provided.\n  description (Optional[str]): Medical condition description to search; optional if code is provided.\n  max_results (int): Maximum number of code records to return; default 10, up to 50.\nReturns:\n  result (object): JSON object with status, search_term, total_results, and matching ICD-10 code records."
  },
  {
    "function_name": "HEALTHCARE_MCP_pubmed_search",
    "default_arguments": {
      "max_results": 5,
      "open_access": False
    },
    "function_comment": "Search medical literature in the PubMed database.\nArgs:\n  query (str): Search query for medical literature.\n  max_results (int): Maximum number of results to return; default 5, range 1-100.\n  date_range (str): Optional year-window filter such as \"5\" for the last 5 years.\n  open_access (bool): Whether to restrict results to open-access articles; default false.\nReturns:\n  result (object): JSON object with status, query, total_results, and matching article records."
  },
  {
    "function_name": "METMUSEUM_MCP_list_departments",
    "default_arguments": {},
    "function_comment": "Lists all the valid departments at The Met"
  },
  {
    "function_name": "NASA_MCP_get_notifications",
    "default_arguments": {
      "notification_type": "all"
    },
    "function_comment": "Retrieve real-time space weather alerts from NASA DONKI Notifications API.\nEndpoint:\n    GET https://api.nasa.gov/DONKI/notifications\n\nPurpose:\n    Provides notification messages for various space weather disturbances,\n    useful for monitoring solar events that may impact satellites, aviation,\n    power grids, or auroras on Earth.\n\nArgs:\n    start_date (str):\n        Filter notifications occurring on/after this date.\n        Format: \"YYYY-MM-DD\".\n        Default: 7 days before current date.\n    end_date (str):\n        Filter notifications occurring on/before this date.\n        Format: \"YYYY-MM-DD\".\n        Default: current date.\n    notification_type (str):\n        Category filter:\n            - \"all\" (default): return every type available\n            - \"FLR\": Solar Flare\n            - \"SEP\": Solar Energetic Particle Event\n            - \"CME\": Coronal Mass Ejection\n            - \"IPS\": Interplanetary Shock\n            - \"MPC\": Magnetopause Crossing\n            - \"GST\": Geomagnetic Storm\n            - \"RBE\": Radiation Belt Enhancement\n            - \"report\": Composite space weather summary reports\n\nReturns:\n    str: Human-readable formatted output including:\n        - Total notification count\n        - For up to 10 notifications:\n          * messageID (unique identifier)\n          * messageType\n          * messageIssueTime\n          * messageHeader\n          * messageBody (first 200 characters)\n\nCommon Errors:\n    - 400 Bad Request: invalid date format / type filter\n    - 429 Too Many Requests: API key rate limit exceeded\n    - 500 Server Errors: NASA endpoint temporarily unavailable\n\nUsage example (MCP prompt):\n    Step 1:\n    Call nasa-mcp/get_notifications with:\n    {\n      \"start_date\": \"2025-10-22\",\n      \"end_date\": \"2025-10-29\",\n      \"notification_type\": \"FLR\"\n    }"
  },
  {
    "function_name": "NATIONALPARKS_getAlerts",
    "default_arguments": {},
    "function_comment": "Get current alerts for national parks.\n  Args:\n    (see schema): Use GetAlertsSchema fields such as park codes or state.\n  Returns:\n    result (text): JSON string listing active alerts."
  },
  {
    "function_name": "NATIONALPARKS_getCampgrounds",
    "default_arguments": {},
    "function_comment": "Get information about available campgrounds and amenities.\n  Args:\n    (see schema): Use GetCampgroundsSchema fields such as park code or state.\n  Returns:\n    result (text): JSON string listing campgrounds."
  },
  {
    "function_name": "NIXOS_nixos_search",
    "default_arguments": {},
    "function_comment": "Search NixOS packages, options, programs, or flakes.\nArgs:\n  query (str): Search term to match.\n  search_type (str): One of packages, options, programs, or flakes.\n  limit (int): Maximum number of results to return (1–100).\n  channel (str): NixOS channel name such as unstable or stable.\nReturns:\n  result (str): Plain-text summary of matching entries or an error message."
  },
  {
    "function_name": "OKX_get_price",
    "default_arguments": {},
    "function_comment": "Get the latest price for an OKX trading instrument.\n  Args:\n    instrument (string): Instrument ID such as BTC-USDT.\n  Returns:\n    result (text): JSON string with current price and basic ticker fields."
  },
  {
    "function_name": "REDDIT_MCP_SERVER_search_hot_posts",
    "default_arguments": {},
    "function_comment": "Retrieve hot posts from a subreddit using the Reddit API.\nArgs:\n  subreddit (str): Subreddit name without the r/ prefix.\n  limit (int): Number of posts to retrieve.\nReturns:\n  result (str): Plain-text list of posts with title, score, comments, author, and link, or an error message."
  },
  {
    "function_name": "TMDB_search_movies",
    "default_arguments": {
      "page": 1
    },
    "function_comment": "Search for movies using The Movie Database API.\n  Args:\n    query (string): Search query text.\n    year (number): Optional filter by release year.\n    page (number): Optional page number, defaults to 1.\n  Returns:\n    result (text): JSON string of the TMDB /search/movie response."
  },
  {
    "function_name": "WEATHER_get_weather",
    "default_arguments": {},
    "function_comment": "Get current weather for a U.S. location using Open‑Meteo with ZIP/city geocoding.\nArgs:\n  location (str): City/state name or a 5‑digit U.S. ZIP code.\n  units (Literal[\"us\",\"metric\"]): \"us\" for imperial units or \"metric\" for SI units.\nReturns:\n  result (str): Plain-text sentence summarizing temperature, wind, code, and place name."
  },
  {
    "function_name": "WIKI_search",
    "default_arguments": {},
    "function_comment": "Search Wikipedia for pages matching a query and return top titles.\nArgs:\n  query (str): Free-text search query.\n  n (int): Number of results to include (1–10).\nReturns:\n  result (str): Plain-text list of up to n titles or a no-results message."
  },
  {
    "function_name": "WIKI_summary",
    "default_arguments": {},
    "function_comment": "Fetch a short Wikipedia summary for the given page title.\nArgs:\n  title (str): Exact or redirectable page title.\nReturns:\n  result (str): Plain-text block with title, short description, and extract."
  }
]
