package uploader

import "testing"

func TestPluginFilePathFromURL(t *testing.T) {
	tests := []struct {
		name    string
		fileURL string
		subPath string
		want    string
	}{
		{
			name:    "direct avatar path",
			fileURL: "https://cdn.example.com/avatar/abc123.png",
			subPath: "avatar",
			want:    "avatar/abc123.png",
		},
		{
			name:    "bucket prefix before avatar path",
			fileURL: "https://cdn.example.com/public-bucket/avatar/abc123.png",
			subPath: "avatar",
			want:    "avatar/abc123.png",
		},
		{
			name:    "missing sub path",
			fileURL: "https://cdn.example.com/post/abc123.png",
			subPath: "avatar",
			want:    "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := pluginFilePathFromURL(tt.fileURL, tt.subPath)
			if got != tt.want {
				t.Fatalf("pluginFilePathFromURL() = %q, want %q", got, tt.want)
			}
		})
	}
}
