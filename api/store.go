package main

import "sync"

// RunTrace holds the raw JSON payload reveived from the Python runner
type RunTrace struct {
	RunID string         `json:"run_id"`
	Data  map[string]any `json:"data"`
}

// Store is a thread-safe in-memory map of run_id -> RunTrace
type Store struct {
	mu     sync.RWMutex
	traces map[string]RunTrace
}

func NewStore() *Store {
	return &Store{traces: make(map[string]RunTrace)}
}

func (s *Store) Save(trace RunTrace) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.traces[trace.RunID] = trace
}

func (s *Store) Get(runID string) (RunTrace, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	trace, ok := s.traces[runID]
	return trace, ok
}

func (s *Store) All() []RunTrace {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make([]RunTrace, 0, len(s.traces))
	for _, t := range s.traces {
		result = append(result, t)
	}
	return result
}
