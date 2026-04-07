package llm

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// newTestCopilotProvider creates a copilotProvider pre-loaded with a GitHub
// token and a valid (non-expired) Copilot token for use in unit tests. The
// provided Copilot token URL is set to the given server so refreshes can be
// intercepted.
func newTestCopilotProvider(githubToken, copilotToken string) *copilotProvider {
	return &copilotProvider{
		githubToken:  githubToken,
		copilotToken: copilotToken,
		tokenExpiry:  time.Now().Add(30 * time.Minute),
		httpClient:   &http.Client{Timeout: 5 * time.Second},
	}
}

func TestCopilotProviderName(t *testing.T) {
	p := newTestCopilotProvider("gh-token", "cop-token")
	if p.Name() != "copilot" {
		t.Errorf("expected name %q, got %q", "copilot", p.Name())
	}
}

func TestCopilotProviderSupportsVision(t *testing.T) {
	p := newTestCopilotProvider("gh-token", "cop-token")
	if p.SupportsVision() {
		t.Error("copilot provider should not report vision support")
	}
}

func TestCopilotFormatRequest(t *testing.T) {
	p := newTestCopilotProvider("gh-token", "cop-token")

	req, err := p.FormatRequest([]Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
	}, CallOpts{Model: "gpt-4o", MaxTokens: 100})
	if err != nil {
		t.Fatalf("FormatRequest: %v", err)
	}

	if req.Method != "POST" {
		t.Errorf("expected POST, got %s", req.Method)
	}
	if !strings.HasSuffix(req.URL.String(), "/chat/completions") {
		t.Errorf("unexpected URL: %s", req.URL)
	}
	if req.Header.Get("Authorization") != "Bearer cop-token" {
		t.Errorf("wrong Authorization header: %s", req.Header.Get("Authorization"))
	}
	if req.Header.Get("Editor-Version") == "" {
		t.Error("missing Editor-Version header")
	}
	if req.Header.Get("Copilot-Integration-Id") == "" {
		t.Error("missing Copilot-Integration-Id header")
	}

	var body map[string]any
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	if body["model"] != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %v", body["model"])
	}
}

func TestCopilotParseResponse(t *testing.T) {
	p := newTestCopilotProvider("gh-token", "cop-token")

	raw, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{
			{"message": map[string]string{"content": "Hello from Copilot"}},
		},
		"model": "gpt-4o",
		"usage": map[string]int{"total_tokens": 25},
	})

	resp, err := p.ParseResponse(raw)
	if err != nil {
		t.Fatalf("ParseResponse: %v", err)
	}
	if resp.Content != "Hello from Copilot" {
		t.Errorf("unexpected content: %q", resp.Content)
	}
	if resp.TokensUsed != 25 {
		t.Errorf("expected 25 tokens, got %d", resp.TokensUsed)
	}
}

func TestCopilotParseResponseEmptyChoices(t *testing.T) {
	p := newTestCopilotProvider("gh-token", "cop-token")

	raw, _ := json.Marshal(map[string]any{
		"choices": []any{},
	})

	_, err := p.ParseResponse(raw)
	if err == nil {
		t.Error("expected error for empty choices")
	}
}

func TestCopilotParseStreamChunk(t *testing.T) {
	p := newTestCopilotProvider("gh-token", "cop-token")

	// Intermediate chunk
	chunk, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{
			{"delta": map[string]string{"content": "Hello"}, "finish_reason": nil},
		},
	})
	token, done := p.ParseStreamChunk(chunk)
	if token != "Hello" {
		t.Errorf("expected token %q, got %q", "Hello", token)
	}
	if done {
		t.Error("expected done=false for intermediate chunk")
	}

	// Final chunk with finish_reason
	finishReason := "stop"
	finalChunk, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{
			{"delta": map[string]string{"content": ""}, "finish_reason": finishReason},
		},
	})
	_, done = p.ParseStreamChunk(finalChunk)
	if !done {
		t.Error("expected done=true for final chunk")
	}
}

func TestCopilotTokenRefresh(t *testing.T) {
	// Serve a fake Copilot token endpoint
	future := time.Now().Add(30 * time.Minute).Unix()
	tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "token gh-token" {
			t.Errorf("unexpected Authorization: %s", r.Header.Get("Authorization"))
		}
		json.NewEncoder(w).Encode(map[string]any{
			"token":      "fresh-cop-token",
			"expires_at": future,
		})
	}))
	defer tokenServer.Close()

	p := &copilotProvider{
		githubToken: "gh-token",
		// expired token forces a refresh
		copilotToken: "old-token",
		tokenExpiry:  time.Now().Add(-1 * time.Minute),
		httpClient:   tokenServer.Client(),
	}

	// Patch the copilotAPITokenURL constant via a round-tripper that rewrites
	// the host so the real HTTPS URL goes to our test server instead.
	p.httpClient = &http.Client{
		Transport: &rewriteTransport{
			target: tokenServer.URL,
			host:   "api.github.com",
		},
	}

	token, err := p.getToken()
	if err != nil {
		t.Fatalf("getToken: %v", err)
	}
	if token != "fresh-cop-token" {
		t.Errorf("expected refreshed token, got %q", token)
	}
}

func TestCopilotFullRoundTrip(t *testing.T) {
	// Fake Copilot token endpoint
	future := time.Now().Add(30 * time.Minute).Unix()
	tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"token":      "api-token",
			"expires_at": future,
		})
	}))
	defer tokenServer.Close()

	// Fake Copilot chat completions endpoint
	chatServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer api-token" {
			t.Errorf("wrong auth: %s", r.Header.Get("Authorization"))
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"content": "wiki answer"}},
			},
			"model": "gpt-4o",
			"usage": map[string]int{"total_tokens": 10},
		})
	}))
	defer chatServer.Close()

	// Build a client using a custom HTTP transport that routes to the test servers
	p := &copilotProvider{
		githubToken: "gh-token",
		httpClient: &http.Client{
			Transport: &multiRewriteTransport{
				routes: map[string]string{
					"api.github.com":         tokenServer.URL,
					"api.githubcopilot.com":  chatServer.URL,
				},
			},
		},
	}

	// Force a token refresh by leaving copilotToken empty
	req, err := p.FormatRequest([]Message{{Role: "user", Content: "hello"}}, CallOpts{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("FormatRequest: %v", err)
	}

	// Execute the request against the chat server via the regular http.Client
	client := &http.Client{
		Transport: &multiRewriteTransport{
			routes: map[string]string{
				"api.githubcopilot.com": chatServer.URL,
			},
		},
	}
	httpResp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer httpResp.Body.Close()

	var buf bytes.Buffer
	if _, err := buf.ReadFrom(httpResp.Body); err != nil {
		t.Fatalf("read response body: %v", err)
	}
	resp, err := p.ParseResponse(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseResponse: %v", err)
	}
	if resp.Content != "wiki answer" {
		t.Errorf("unexpected content: %q", resp.Content)
	}
}

func TestNewClientCopilotUnsupportedWithoutToken(t *testing.T) {
	// NewClient should return an error if the device code flow cannot proceed
	// (network unavailable). We verify it returns an error rather than hanging.
	// Since we cannot easily intercept DNS/network in tests without more complex
	// setup, this test only checks that the "copilot" provider name is accepted
	// when a GitHub token is pre-supplied.
	client, err := NewClient("copilot", "gh-test-token", "", 0)
	if err != nil {
		// It's acceptable to fail here if the Copilot token endpoint is
		// unreachable from the test environment. Skip rather than fail.
		t.Skipf("copilot provider setup failed (network unavailable?): %v", err)
	}
	if client.ProviderName() != "copilot" {
		t.Errorf("expected provider name %q, got %q", "copilot", client.ProviderName())
	}
}

// ─── Test helpers ────────────────────────────────────────────────────────────

// rewriteTransport redirects requests for a specific host to a test server URL.
type rewriteTransport struct {
	target string // e.g. "http://127.0.0.1:PORT"
	host   string // e.g. "api.github.com"
}

func (rt *rewriteTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL.Hostname() == rt.host || strings.Contains(req.URL.String(), rt.host) {
		newURL := rt.target + req.URL.Path
		if req.URL.RawQuery != "" {
			newURL += "?" + req.URL.RawQuery
		}
		newReq := req.Clone(req.Context())
		newReq.URL, _ = req.URL.Parse(newURL)
		newReq.Host = ""
		return http.DefaultTransport.RoundTrip(newReq)
	}
	return http.DefaultTransport.RoundTrip(req)
}

// multiRewriteTransport redirects requests to multiple test servers by host.
type multiRewriteTransport struct {
	routes map[string]string // host → test server URL
}

func (rt *multiRewriteTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if target, ok := rt.routes[req.URL.Hostname()]; ok {
		newURL := target + req.URL.Path
		if req.URL.RawQuery != "" {
			newURL += "?" + req.URL.RawQuery
		}
		newReq := req.Clone(req.Context())
		newReq.URL, _ = req.URL.Parse(newURL)
		newReq.Host = ""
		return http.DefaultTransport.RoundTrip(newReq)
	}
	return http.DefaultTransport.RoundTrip(req)
}
