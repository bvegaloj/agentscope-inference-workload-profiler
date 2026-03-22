package main

import (
	"encoding/json"
	"net/http"
)

func PostTrace(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var trace RunTrace
		if err := json.NewDecoder(r.Body).Decode(&trace); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		if trace.RunID == "" {
			http.Error(w, "run_id is required", http.StatusBadRequest)
			return
		}
		store.Save(trace)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]string{"run_id": trace.RunID})
	}
}

func GetTrace(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		runID := r.PathValue("run_id")
		trace, ok := store.Get(runID)
		if !ok {
			http.Error(w, "trace not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(trace)
	}
}

func GetAllTraces(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		traces := store.All()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(traces)
	}
}
