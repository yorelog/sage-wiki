package llm

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

// errWriter is the destination for interactive device-code messages. It is a
// package-level variable so tests can redirect the output.
var errWriter = io.Writer(os.Stderr)

const (
	copilotClientID     = "Iv1.b507a08c87ecfe98"
	copilotDeviceURL    = "https://github.com/login/device/code"
	copilotOAuthURL     = "https://github.com/login/oauth/access_token"
	copilotAPITokenURL  = "https://api.github.com/copilot_internal/v2/token"
	copilotBaseURL      = "https://api.githubcopilot.com"
	copilotEditorVer    = "vscode/1.97.0"
	copilotPluginVer    = "copilot-chat/0.24.0"
	copilotIntegration  = "vscode-chat"
)

// copilotProvider implements the GitHub Copilot API using a GitHub OAuth token
// for authentication. If no token is supplied, the GitHub device code flow is
// triggered interactively so the user can authorise the application in a browser.
type copilotProvider struct {
	githubToken  string // long-lived GitHub OAuth token
	copilotToken string // short-lived Copilot API token
	tokenExpiry  time.Time
	mu           sync.Mutex
	httpClient   *http.Client
}

// newCopilotProvider creates a copilotProvider. When apiKey is empty it runs the
// GitHub device code authorisation flow (prints instructions to stderr and polls
// until the user completes the browser step). When apiKey is non-empty it is
// treated as a GitHub OAuth token and no browser interaction is required.
func newCopilotProvider(apiKey string) (*copilotProvider, error) {
	p := &copilotProvider{
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}

	if apiKey == "" {
		token, err := p.runDeviceCodeFlow()
		if err != nil {
			return nil, fmt.Errorf("copilot: device code auth: %w", err)
		}
		p.githubToken = token
	} else {
		p.githubToken = apiKey
	}

	return p, nil
}

func (p *copilotProvider) Name() string         { return "copilot" }
func (p *copilotProvider) SupportsVision() bool { return false }

// getToken returns a valid Copilot API token, refreshing it if necessary.
func (p *copilotProvider) getToken() (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.copilotToken != "" && time.Now().Before(p.tokenExpiry.Add(-30*time.Second)) {
		return p.copilotToken, nil
	}

	return p.refreshCopilotToken()
}

// refreshCopilotToken exchanges the GitHub OAuth token for a short-lived
// Copilot API token. Must be called with p.mu held.
func (p *copilotProvider) refreshCopilotToken() (string, error) {
	req, err := http.NewRequest("GET", copilotAPITokenURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "token "+p.githubToken)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Editor-Version", copilotEditorVer)
	req.Header.Set("Editor-Plugin-Version", copilotPluginVer)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch copilot token: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("fetch copilot token: HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Token     string `json:"token"`
		ExpiresAt int64  `json:"expires_at"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("parse copilot token: %w", err)
	}
	if result.Token == "" {
		return "", fmt.Errorf("copilot token response missing token field")
	}

	p.copilotToken = result.Token
	p.tokenExpiry = time.Unix(result.ExpiresAt, 0)
	return p.copilotToken, nil
}

// formatBody builds an OpenAI-compatible request body for the Copilot API.
func (p *copilotProvider) formatBody(messages []Message, opts CallOpts, stream bool) map[string]any {
	var apiMessages []map[string]string
	for _, m := range messages {
		apiMessages = append(apiMessages, map[string]string{
			"role":    m.Role,
			"content": m.Content,
		})
	}

	body := map[string]any{
		"model":    opts.Model,
		"messages": apiMessages,
	}
	if stream {
		body["stream"] = true
	}
	if opts.MaxTokens > 0 {
		body["max_tokens"] = opts.MaxTokens
	}
	if opts.Temperature > 0 {
		body["temperature"] = opts.Temperature
	}
	return body
}

func (p *copilotProvider) makeRequest(body map[string]any) (*http.Request, error) {
	token, err := p.getToken()
	if err != nil {
		return nil, fmt.Errorf("copilot: get token: %w", err)
	}

	req, err := http.NewRequest("POST", copilotBaseURL+"/chat/completions", jsonBody(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Editor-Version", copilotEditorVer)
	req.Header.Set("Editor-Plugin-Version", copilotPluginVer)
	req.Header.Set("Copilot-Integration-Id", copilotIntegration)
	return req, nil
}

// FormatRequest implements Provider.
func (p *copilotProvider) FormatRequest(messages []Message, opts CallOpts) (*http.Request, error) {
	return p.makeRequest(p.formatBody(messages, opts, false))
}

// FormatStreamRequest implements StreamingProvider.
func (p *copilotProvider) FormatStreamRequest(messages []Message, opts CallOpts) (*http.Request, error) {
	return p.makeRequest(p.formatBody(messages, opts, true))
}

// ParseStreamChunk implements StreamingProvider (OpenAI SSE format).
func (p *copilotProvider) ParseStreamChunk(data []byte) (string, bool) {
	var chunk struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &chunk); err != nil {
		return "", false
	}
	if len(chunk.Choices) == 0 {
		return "", false
	}
	done := chunk.Choices[0].FinishReason != nil
	return chunk.Choices[0].Delta.Content, done
}

// ParseResponse implements Provider (OpenAI response format).
func (p *copilotProvider) ParseResponse(body []byte) (*Response, error) {
	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Model string `json:"model"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("copilot: parse response: %w", err)
	}
	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("copilot: empty choices in response")
	}
	return &Response{
		Content:    result.Choices[0].Message.Content,
		Model:      result.Model,
		TokensUsed: result.Usage.TotalTokens,
	}, nil
}

// ─── Device Code Auth ────────────────────────────────────────────────────────

// deviceCodeResponse holds the initial device code response from GitHub.
type deviceCodeResponse struct {
	DeviceCode      string `json:"device_code"`
	UserCode        string `json:"user_code"`
	VerificationURI string `json:"verification_uri"`
	ExpiresIn       int    `json:"expires_in"`
	Interval        int    `json:"interval"`
}

// runDeviceCodeFlow performs the GitHub OAuth device code flow and returns the
// resulting GitHub OAuth access token. Instructions and the user code are
// printed to stderr so they do not pollute stdout/MCP transports.
func (p *copilotProvider) runDeviceCodeFlow() (string, error) {
	dc, err := p.requestDeviceCode()
	if err != nil {
		return "", err
	}

	fmt.Fprintf(errWriter, "\nGitHub Copilot authentication required.\n  1. Open: %s\n  2. Enter code: %s\n\n",
		dc.VerificationURI, dc.UserCode)

	return p.pollForToken(dc)
}

// requestDeviceCode calls the GitHub device code endpoint.
func (p *copilotProvider) requestDeviceCode() (*deviceCodeResponse, error) {
	data := url.Values{
		"client_id": {copilotClientID},
		"scope":     {"read:user"},
	}
	req, err := http.NewRequest("POST", copilotDeviceURL, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request device code: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("request device code: HTTP %d: %s", resp.StatusCode, string(body))
	}

	var dc deviceCodeResponse
	if err := json.Unmarshal(body, &dc); err != nil {
		return nil, fmt.Errorf("parse device code response: %w", err)
	}
	if dc.DeviceCode == "" {
		return nil, fmt.Errorf("device code response missing device_code")
	}
	if dc.Interval <= 0 {
		dc.Interval = 5
	}
	return &dc, nil
}

// pollForToken polls GitHub's OAuth token endpoint until the user completes
// the browser authorisation or the device code expires.
func (p *copilotProvider) pollForToken(dc *deviceCodeResponse) (string, error) {
	deadline := time.Now().Add(time.Duration(dc.ExpiresIn) * time.Second)
	interval := time.Duration(dc.Interval) * time.Second

	for time.Now().Before(deadline) {
		time.Sleep(interval)

		token, retry, err := p.tryExchangeToken(dc.DeviceCode)
		if err != nil {
			return "", err
		}
		if retry {
			continue
		}
		return token, nil
	}

	return "", fmt.Errorf("device code expired before authorisation was completed")
}

// tryExchangeToken attempts a single token exchange. It returns (token, false,
// nil) on success, ("", true, nil) when the user hasn't authorised yet (slow
// down / authorisation_pending), and ("", false, err) on a terminal error.
func (p *copilotProvider) tryExchangeToken(deviceCode string) (string, bool, error) {
	data := url.Values{
		"client_id":   {copilotClientID},
		"device_code": {deviceCode},
		"grant_type":  {"urn:ietf:params:oauth:grant-type:device_code"},
	}
	req, err := http.NewRequest("POST", copilotOAuthURL, strings.NewReader(data.Encode()))
	if err != nil {
		return "", false, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return "", false, fmt.Errorf("poll token: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var result struct {
		AccessToken string `json:"access_token"`
		Error       string `json:"error"`
		Interval    int    `json:"interval"` // slow_down may increase interval
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", false, fmt.Errorf("parse token response: %w", err)
	}

	switch result.Error {
	case "":
		if result.AccessToken == "" {
			return "", false, fmt.Errorf("token response missing access_token")
		}
		return result.AccessToken, false, nil
	case "authorization_pending":
		return "", true, nil
	case "slow_down":
		// GitHub asked us to back off; the response includes a new interval
		return "", true, nil
	case "expired_token":
		return "", false, fmt.Errorf("device code expired")
	case "access_denied":
		return "", false, fmt.Errorf("authorisation denied by user")
	default:
		return "", false, fmt.Errorf("token exchange error: %s", result.Error)
	}
}
