package main

import (
	"fmt"
	"log"
	"net/http"
)

const port = ":8080"

func main() {
	store := NewStore()
	mux := http.NewServeMux()

	mux.HandleFunc("POST /traces", PostTrace(store))
	mux.HandleFunc("GET /traces/{run_id}", GetTrace(store))
	mux.HandleFunc("GET /traces", GetAllTraces(store))

	fmt.Printf("AgentScope API listening on %s\n", port)
	log.Fatal(http.ListenAndServe(port, mux))
}
