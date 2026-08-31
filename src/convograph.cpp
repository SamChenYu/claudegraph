// convograph — the real conversation graph charts itself (nodes fly in and
// settle), then the app assembles around it into an interactive graph of all
// your Claude Code conversations. Nodes are sessions (sized by prompt count,
// colored by project); edges connect *related* conversations (shared
// vocabulary via TF-IDF cosine similarity, plus same-project affinity). The
// graph settles with a live force-directed layout you can click, hover and drag.
//
// Data source (read-only): ~/.claude/history.jsonl for prompts/projects/times,
// and ~/.claude/projects/<cwd>/<session>.jsonl for the auto-generated titles.
//
// Controls:
//   click node   select + show details        drag node   reposition
//   1-6          jump to a related conversation  Tab       cycle selection
//   r            reheat the layout               space     pause/resume
//   Esc          deselect                        q         quit
//
// Build: see Makefile.

#include <ftxui/component/component.hpp>
#include <ftxui/component/component_base.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/mouse.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/canvas.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/box.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "json.hpp"

using namespace ftxui;
namespace fs = std::filesystem;

// ----------------------------------------------------------------------------
// Model
// ----------------------------------------------------------------------------
struct Conv {
  std::string id, title, project, projName;
  std::string txPath;  // transcript .jsonl, for lazy-loading full chat history
  int prompts = 0;
  std::string tFirst, tLast;
  std::vector<std::string> texts;               // prompt strings + title
  std::unordered_map<std::string, float> vec;   // L2-normalized tf-idf
  float x = 0, y = 0;                            // world position
  float sx = 0, sy = 0, sr = 3;                  // screen (canvas-dot) position
  int hue = 0;
};
struct Edge {
  int a, b;
  float w;
};
struct Graph {
  std::vector<Conv> nodes;
  std::vector<Edge> edges;
  std::vector<std::vector<std::pair<int, float>>> adj;  // node -> (nbr, weight)
  int nprojects = 0;
};

// ----------------------------------------------------------------------------
// Text helpers
// ----------------------------------------------------------------------------
static const std::unordered_set<std::string>& stopwords() {
  static const std::unordered_set<std::string> s = {
      "the",  "and",  "for",  "you",  "that", "this", "with", "can",  "are",
      "not",  "but",  "have", "was",  "your", "get",  "how",  "все",  "add",
      "use",  "make", "would","like", "some", "then", "them", "into", "just",
      "want", "need", "should","could","from", "when", "what", "why",  "does",
      "did",  "will", "out",  "one",  "all",  "any",  "its",  "here", "there",
      "also", "now",  "new",  "let",  "see",  "run",  "set",  "way",
  };
  return s;
}
static std::vector<std::string> tokenize(const std::string& in) {
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&] {
    if (cur.size() >= 3 && !stopwords().count(cur)) {
      bool alldigit = true;
      for (char c : cur)
        if (c < '0' || c > '9')
          alldigit = false;
      if (!alldigit)
        out.push_back(cur);
    }
    cur.clear();
  };
  for (char c : in) {
    if (c >= 'A' && c <= 'Z')
      c = char(c - 'A' + 'a');
    if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))
      cur += c;
    else
      flush();
  }
  flush();
  return out;
}
static int hue_of(const std::string& s) {
  uint32_t h = 2166136261u;
  for (char c : s) {
    h ^= (uint8_t)c;
    h *= 16777619u;
  }
  return int(h % 256);
}
static std::string basename_of(const std::string& p) {
  auto pos = p.find_last_of('/');
  std::string b = (pos == std::string::npos) ? p : p.substr(pos + 1);
  return b.empty() ? p : b;
}

// ----------------------------------------------------------------------------
// Loading
// ----------------------------------------------------------------------------
static std::string read_title(const std::string& path) {
  std::ifstream f(path);
  if (!f)
    return "";
  std::string line, title;
  while (std::getline(f, line)) {
    if (line.find("\"aiTitle\"") == std::string::npos)
      continue;
    mjson::Value v;
    if (mjson::parse_line(line, v)) {
      if (auto* t = v.find("aiTitle"))
        title = t->as_str(title);  // keep the last (most-refined) title
    }
  }
  return title;
}

// A single conversation turn for the chat-history view.
struct Turn {
  int role = 0;  // 0 = you, 1 = Claude
  std::string text;
  std::string ts;
};

// Parse a transcript .jsonl into readable turns: your prompts, and Claude's
// replies (its visible text plus a compact marker for each tool it ran; internal
// "thinking" and raw tool-result payloads are omitted). Loaded lazily, per node.
static std::vector<Turn> load_history(const std::string& path) {
  std::vector<Turn> out;
  std::ifstream f(path);
  if (!f)
    return out;
  std::string line;
  while (std::getline(f, line)) {
    // Cheap pre-filter: only user/assistant turns carry a message we render.
    if (line.find("\"user\"") == std::string::npos &&
        line.find("\"assistant\"") == std::string::npos)
      continue;
    mjson::Value v;
    if (!mjson::parse_line(line, v) || v.type != mjson::Value::Obj)
      continue;
    auto* ty = v.find("type");
    if (!ty)
      continue;
    std::string t = ty->as_str();
    if (t != "user" && t != "assistant")
      continue;
    auto* msg = v.find("message");
    if (!msg)
      continue;
    auto* content = msg->find("content");
    if (!content)
      continue;
    std::string ts;
    if (auto* p = v.find("timestamp"))
      ts = p->as_str();

    if (t == "user") {
      // Real prompts are plain strings; list content is tool results — skip it.
      if (content->type == mjson::Value::Str && !content->str.empty())
        out.push_back({0, content->str, ts});
      continue;
    }
    // assistant: gather text blocks and tool-use markers, in order.
    std::string body;
    auto add = [&](const std::string& s) {
      if (s.empty())
        return;
      if (!body.empty())
        body += "\n\n";
      body += s;
    };
    if (content->type == mjson::Value::Arr) {
      for (auto& it : content->arr) {
        if (it.type != mjson::Value::Obj)
          continue;
        auto* k = it.find("type");
        if (!k)
          continue;
        std::string kind = k->as_str();
        if (kind == "text") {
          if (auto* tx = it.find("text"))
            add(tx->as_str());
        } else if (kind == "tool_use") {
          std::string nm;
          if (auto* n = it.find("name"))
            nm = n->as_str();
          add("· ran " + (nm.empty() ? "a tool" : nm));
        }
      }
    } else if (content->type == mjson::Value::Str) {
      add(content->str);
    }
    if (!body.empty())
      out.push_back({1, body, ts});
  }
  return out;
}

static Graph load_graph() {
  Graph g;
  const char* home = std::getenv("HOME");
  if (!home)
    return g;
  std::string root = std::string(home) + "/.claude";
  std::string history = root + "/history.jsonl";

  // 1) Accumulate sessions from history.jsonl.
  std::unordered_map<std::string, int> idx;  // sessionId -> node index
  {
    std::ifstream f(history);
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty())
        continue;
      mjson::Value v;
      if (!mjson::parse_line(line, v) || v.type != mjson::Value::Obj)
        continue;
      auto* sid = v.find("sessionId");
      if (!sid)
        continue;
      std::string id = sid->as_str();
      if (id.empty())
        continue;
      auto it = idx.find(id);
      Conv* c;
      if (it == idx.end()) {
        idx[id] = (int)g.nodes.size();
        g.nodes.emplace_back();
        c = &g.nodes.back();
        c->id = id;
        if (auto* pr = v.find("project"))
          c->project = pr->as_str();
        c->projName = basename_of(c->project);
        c->hue = hue_of(c->projName);
      } else {
        c = &g.nodes[it->second];
      }
      c->prompts++;
      std::string disp, ts;
      if (auto* d = v.find("display"))
        disp = d->as_str();
      if (auto* t = v.find("timestamp"))
        ts = t->as_str();
      if (!disp.empty())
        c->texts.push_back(disp);
      if (!ts.empty()) {
        if (c->tFirst.empty() || ts < c->tFirst)
          c->tFirst = ts;
        if (ts > c->tLast)
          c->tLast = ts;
      }
    }
  }
  if (g.nodes.empty())
    return g;

  // 2) Map sessionId -> transcript file, then pull the title.
  std::unordered_map<std::string, std::string> tx;  // id -> path
  std::error_code ec;
  fs::path pdir = root + "/projects";
  if (fs::exists(pdir, ec)) {
    for (auto& proj : fs::directory_iterator(pdir, ec)) {
      if (!proj.is_directory())
        continue;
      for (auto& file : fs::directory_iterator(proj.path(), ec)) {
        if (file.path().extension() == ".jsonl")
          tx[file.path().stem().string()] = file.path().string();
      }
    }
  }
  for (auto& c : g.nodes) {
    auto it = tx.find(c.id);
    if (it != tx.end()) {
      c.txPath = it->second;
      c.title = read_title(it->second);
    }
    if (c.title.empty()) {  // fallback: first prompt, trimmed
      std::string t = c.texts.empty() ? "(untitled)" : c.texts.front();
      for (char& ch : t)
        if (ch == '\n' || ch == '\r')
          ch = ' ';
      if (t.size() > 46)
        t = t.substr(0, 44) + "…";
      c.title = t;
    }
  }

  // 3) TF-IDF vectors.
  int N = (int)g.nodes.size();
  std::unordered_map<std::string, int> df;
  std::vector<std::unordered_map<std::string, int>> tf(N);
  for (int i = 0; i < N; i++) {
    std::unordered_set<std::string> seen;
    auto feed = [&](const std::string& s) {
      for (auto& w : tokenize(s)) {
        tf[i][w]++;
        seen.insert(w);
      }
    };
    feed(g.nodes[i].title);
    for (auto& t : g.nodes[i].texts)
      feed(t);
    for (auto& w : seen)
      df[w]++;
  }
  for (int i = 0; i < N; i++) {
    double norm = 0;
    auto& vec = g.nodes[i].vec;
    for (auto& [w, c] : tf[i]) {
      float idf = std::log((N + 1.0) / (df[w] + 1.0)) + 1.0;
      float wt = c * idf;
      vec[w] = wt;
      norm += double(wt) * wt;
    }
    norm = std::sqrt(norm);
    if (norm > 1e-9)
      for (auto& [w, wt] : vec)
        wt = float(wt / norm);
  }

  // 4) Candidate edges: cosine similarity, boosted for same project.
  auto sim = [&](int i, int j) {
    const auto& a = g.nodes[i].vec;
    const auto& b = g.nodes[j].vec;
    const auto& small = a.size() < b.size() ? a : b;
    const auto& big = a.size() < b.size() ? b : a;
    float s = 0;
    for (auto& [w, wt] : small) {
      auto it = big.find(w);
      if (it != big.end())
        s += wt * it->second;
    }
    return s;
  };
  std::vector<std::vector<std::pair<int, float>>> cand(N);
  for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++) {
      float s = sim(i, j);
      bool sameProj =
          !g.nodes[i].project.empty() && g.nodes[i].project == g.nodes[j].project;
      float w = sameProj ? std::max(s, 0.28f) : s;
      if (w > 0.14f) {
        cand[i].push_back({j, w});
        cand[j].push_back({i, w});
      }
    }
  // Keep each node's strongest few links, union into an undirected set.
  std::set<std::pair<int, int>> chosen;
  std::map<std::pair<int, int>, float> wmap;
  for (int i = 0; i < N; i++) {
    auto& v = cand[i];
    std::sort(v.begin(), v.end(),
              [](auto& a, auto& b) { return a.second > b.second; });
    for (int k = 0; k < (int)v.size() && k < 6; k++) {
      int a = std::min(i, v[k].first), b = std::max(i, v[k].first);
      chosen.insert({a, b});
      wmap[{a, b}] = std::max(wmap[{a, b}], v[k].second);
    }
  }
  g.adj.assign(N, {});
  for (auto& e : chosen) {
    float w = wmap[e];
    g.edges.push_back({e.first, e.second, w});
    g.adj[e.first].push_back({e.second, w});
    g.adj[e.second].push_back({e.first, w});
  }

  // 5) Initial positions: pre-cluster by project so the layout settles nicely.
  std::unordered_map<std::string, int> pindex;
  for (auto& c : g.nodes)
    if (!pindex.count(c.projName))
      pindex[c.projName] = (int)pindex.size();
  g.nprojects = (int)pindex.size();
  std::mt19937 rng(1234);
  auto fr = [&](float a, float b) {
    return a + (b - a) * (rng() / float(rng.max()));
  };
  for (auto& c : g.nodes) {
    float base = 2 * 3.14159265f * pindex[c.projName] / std::max(1, g.nprojects);
    float ang = base + fr(-0.5f, 0.5f);
    float r = 30 + fr(0, 14);
    c.x = std::cos(ang) * r;
    c.y = std::sin(ang) * r;
  }
  return g;
}

// ----------------------------------------------------------------------------
// Force-directed layout (Fruchterman-Reingold-ish), one settling iteration.
// ----------------------------------------------------------------------------
static constexpr float kRmax = 95.0f;  // hard bound so nothing escapes the view

static void layout_iter(Graph& g, float temp, int pinned) {
  int n = (int)g.nodes.size();
  if (n == 0)
    return;
  float area = 120.0f * 95.0f;
  float k = 0.62f * std::sqrt(area / n);  // ideal edge length
  std::vector<float> dx(n, 0), dy(n, 0);

  // Repulsion between every pair, plus a hard short-range separation so that
  // cliques of near-identical conversations spread into a readable ring instead
  // of collapsing onto one point.
  const float minsep = 15.0f;
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      float ex = g.nodes[i].x - g.nodes[j].x;
      float ey = g.nodes[i].y - g.nodes[j].y;
      float d = std::sqrt(ex * ex + ey * ey) + 0.01f;
      float f = k * k / d;
      if (d < minsep)
        f += (minsep - d) * 9.0f;  // soft-collision push
      float ux = ex / d, uy = ey / d;
      dx[i] += ux * f; dy[i] += uy * f;
      dx[j] -= ux * f; dy[j] -= uy * f;
    }
  // Attraction along edges.
  for (auto& e : g.edges) {
    float ex = g.nodes[e.a].x - g.nodes[e.b].x;
    float ey = g.nodes[e.a].y - g.nodes[e.b].y;
    float d = std::sqrt(ex * ex + ey * ey) + 0.01f;
    float f = d * d / k * (0.22f + e.w * 0.7f);  // looser: cliques don't crush
    float ux = ex / d, uy = ey / d;
    dx[e.a] -= ux * f; dy[e.a] -= uy * f;
    dx[e.b] += ux * f; dy[e.b] += uy * f;
  }
  // Mild gravity so disconnected nodes drift inward rather than to infinity.
  for (int i = 0; i < n; i++) {
    dx[i] -= g.nodes[i].x * 0.06f;
    dy[i] -= g.nodes[i].y * 0.06f;
  }
  // Integrate with a per-step cap (temperature).
  for (int i = 0; i < n; i++) {
    if (i == pinned)
      continue;
    float d = std::sqrt(dx[i] * dx[i] + dy[i] * dy[i]);
    if (d < 1e-4f)
      continue;
    float m = std::min(d, temp);
    g.nodes[i].x += dx[i] / d * m;
    g.nodes[i].y += dy[i] / d * m;
  }
  // Recenter on the centroid (keeps the whole graph framed) and clamp every
  // node inside a bounded disc so repulsion can never fling one off-screen.
  float mx = 0, my = 0;
  for (auto& v : g.nodes) { mx += v.x; my += v.y; }
  mx /= n; my /= n;
  for (auto& v : g.nodes) {
    v.x -= mx; v.y -= my;
    float r = std::sqrt(v.x * v.x + v.y * v.y);
    if (r > kRmax) { v.x *= kRmax / r; v.y *= kRmax / r; }
  }
}

// ----------------------------------------------------------------------------
// Small utilities
// ----------------------------------------------------------------------------
static std::string trim_ts(const std::string& ts) {
  // "2026-08-30T20:11:33.123Z" -> "2026-08-30 20:11"
  if (ts.size() < 16)
    return ts;
  std::string s = ts.substr(0, 16);
  s[10] = ' ';
  return s;
}
static std::string ellipsis(std::string s, size_t n) {
  for (char& c : s)
    if (c == '\n' || c == '\r')
      c = ' ';
  if (s.size() > n)
    s = s.substr(0, n - 1) + "…";
  return s;
}

// A 2D camera: `zoom` canvas-dots per world unit, centered on (camx, camy).
struct Camera {
  float zoom = 3.0f;
  float camx = 0, camy = 0;
  bool user = false;  // has the user panned/zoomed? (stops auto-fit)
};

// Fit a world-space bounding box into the canvas.
static void camera_fit_bounds(float minx, float maxx, float miny, float maxy,
                              int GW, int GH, Camera& cam) {
  float m = 18;
  float sw = std::max(1.0f, maxx - minx), sh = std::max(1.0f, maxy - miny);
  cam.zoom = std::min((GW - 2 * m) / sw, (GH - 2 * m) / sh);
  cam.zoom = std::clamp(cam.zoom, 0.4f, 24.0f);
  cam.camx = (minx + maxx) / 2;
  cam.camy = (miny + maxy) / 2;
}

// Fit the whole graph into the canvas (used until the user takes the camera).
static void camera_fit(const Graph& g, int GW, int GH, Camera& cam) {
  if (g.nodes.empty())
    return;
  float minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
  for (auto& v : g.nodes) {
    minx = std::min(minx, v.x); maxx = std::max(maxx, v.x);
    miny = std::min(miny, v.y); maxy = std::max(maxy, v.y);
  }
  camera_fit_bounds(minx, maxx, miny, maxy, GW, GH, cam);
}

// Draw the whole graph. Updates each node's screen (canvas-dot) position for
// hit-testing. Declutters: weak edges hidden, non-overlapping labels only, and
// a hard focus-fade when a node is selected. Pure enough to render headlessly.
//
// `alpha`, if non-null, is a per-node reveal multiplier in [0,1] used by the
// intro so the *real* graph can fade itself in as it flies into place; an edge
// shows at the min of its endpoints' alpha. When null, everything is fully lit.
// `pop`, if non-null, is a per-node shutdown progress in [0,1]: 0 = intact,
// rising through a bright expanding burst, 1 = gone. Drives the quit animation.
static Canvas render_graph(Graph& g, int GW, int GH, const Camera& cam,
                           int selected, int hover, float tm,
                           const std::vector<float>* alpha = nullptr,
                           const std::vector<float>* pop = nullptr) {
  auto c = Canvas(GW, GH);
  int n = (int)g.nodes.size();
  if (n == 0)
    return c;

  // Screen position + a gentle per-node "float" so the graph feels alive even
  // at rest. Amplitude is tiny (~1 dot) so it reads as breathing, not chaos.
  for (int i = 0; i < n; i++) {
    auto& v = g.nodes[i];
    float ph = i * 2.399963f;  // golden-angle phase spread
    float bx = 1.2f * std::sin(tm * 0.9f + ph);
    float by = 1.2f * std::cos(tm * 0.8f + ph * 1.3f);
    v.sx = GW / 2.0f + (v.x - cam.camx) * cam.zoom + bx;
    v.sy = GH / 2.0f + (v.y - cam.camy) * cam.zoom + by;
    v.sr = 1.0f;  // small uniform dots — this is a graph, not a cloud
  }

  auto is_nbr = [&](int who, int i) {
    if (who < 0)
      return false;
    for (auto& p : g.adj[who])
      if (p.first == i)
        return true;
    return false;
  };
  // A node is "focused" if nothing is selected, or it's the selection/neighbor.
  auto focused = [&](int i) {
    return selected < 0 || i == selected || is_nbr(selected, i);
  };

  // ---- edges (weak ones hidden; when selected, only its edges show) ----
  const float kWeakHide = 0.26f;
  for (auto& e : g.edges) {
    bool incSel = selected >= 0 && (e.a == selected || e.b == selected);
    bool incHov = hover >= 0 && (e.a == hover || e.b == hover);
    int r, gg, b;
    bool draw;
    if (selected >= 0) {
      draw = incSel;  // ego view: hide everything not touching the selection
      r = 210; gg = 195; b = 120;
    } else if (incHov) {
      draw = true;
      r = 150; gg = 150; b = 180;
    } else {
      draw = e.w >= kWeakHide;
      r = 58; gg = 58; b = 72;
    }
    // During the intro, an edge is only as present as its dimmer endpoint; on
    // shutdown it snaps out with whichever endpoint pops first.
    float ea = alpha ? std::min((*alpha)[e.a], (*alpha)[e.b]) : 1.0f;
    if (pop)
      ea *= std::min(1.0f - (*pop)[e.a], 1.0f - (*pop)[e.b]);
    if (draw && ea > 0.02f)
      c.DrawPointLine(int(g.nodes[e.a].sx), int(g.nodes[e.a].sy),
                      int(g.nodes[e.b].sx), int(g.nodes[e.b].sy),
                      Color::RGB((uint8_t)(r * ea), (uint8_t)(gg * ea),
                                 (uint8_t)(b * ea)));
  }

  // ---- nodes (unfocused ones fade way down; all twinkle slightly) ----
  for (int i = 0; i < n; i++) {
    auto& v = g.nodes[i];
    // Shutdown: each node snaps a hot white flash, throws a shockwave ring that
    // explodes outward, and is gone — quick and springy, not a slow-fading blob.
    if (pop) {
      float e = (*pop)[i];
      if (e >= 1.0f)
        continue;  // already gone
      if (e > 0.0f) {
        // Core spark: peaks instantly, collapses fast (quadratic), whites out at
        // the peak then dies back into the node's hue.
        float core = std::clamp(1.0f - e / 0.40f, 0.0f, 1.0f);
        core *= core;
        if (core > 0.02f) {
          int dotR = (e < 0.14f) ? 2 : 1;  // a single-beat swell, then a point
          int sat = int(std::clamp(45.0f + 175.0f * (1.0f - core), 0.0f, 255.0f));
          int val = int(std::clamp(255.0f * core, 0.0f, 255.0f));
          c.DrawPointCircleFilled(int(v.sx), int(v.sy), dotR,
                                  Color::HSV((uint8_t)v.hue, (uint8_t)sat,
                                             (uint8_t)val));
        }
        // Shockwave ring: launches just after the flash, expands ease-out (fast
        // then decelerating) and fades quadratically — reads as an outward burst.
        float re = std::clamp((e - 0.05f) / 0.95f, 0.0f, 1.0f);
        float eased = 1.0f - (1.0f - re) * (1.0f - re);
        int ringR = int(std::round(1.0f + eased * 5.0f));
        float rf = 1.0f - re;
        int ringV = int(std::clamp(235.0f * rf * rf, 0.0f, 255.0f));
        if (ringR >= 1 && ringV > 8)
          c.DrawPointCircle(int(v.sx), int(v.sy), ringR,
                            Color::HSV((uint8_t)v.hue, 160, (uint8_t)ringV));
        continue;
      }
      // e == 0: fall through and draw the node normally (not yet popped).
    }
    float a = alpha ? (*alpha)[i] : 1.0f;
    if (a <= 0.02f)
      continue;  // not yet revealed during the intro
    bool foc = focused(i);
    float tw = 0.82f + 0.18f * std::sin(tm * 1.7f + i * 2.399963f);  // twinkle
    int sat = foc ? 215 : 40;
    int val = int((foc ? 245 : 70) * tw * a);
    c.DrawPointCircleFilled(int(v.sx), int(v.sy), int(v.sr),
                            Color::HSV((uint8_t)v.hue, (uint8_t)sat,
                                       (uint8_t)std::clamp(val, 0, 255)));
  }
  auto ring = [&](int i, Color col) {
    if (i < 0)
      return;
    c.DrawPointCircle(int(g.nodes[i].sx), int(g.nodes[i].sy),
                      int(g.nodes[i].sr) + 2, col);
  };
  ring(hover, Color::RGB(150, 150, 170));
  ring(selected, Color::RGB(245, 245, 160));

  // ---- label: only the hovered node, so the canvas stays clean ----
  if (hover >= 0) {
    auto& v = g.nodes[hover];
    std::string txt = ellipsis(v.title, 32);
    int len = (int)txt.size();
    int cellsW = GW / 2;
    int lx = int(v.sx + v.sr + 3) / 2;      // cell x
    if (lx + len > cellsW)                    // flip to the left if it overflows
      lx = int(v.sx - v.sr - 3) / 2 - len;
    lx = std::max(0, lx);
    c.DrawText(lx * 2, int(v.sy), txt, Color::RGB(240, 240, 250));
  }
  return c;
}

int main() {
  auto screen = ScreenInteractive::Fullscreen();
  Graph g = load_graph();

  // Settle the *real* graph once to get its final layout. Those settled
  // positions are the "targets" the intro flies the nodes into — so the loading
  // screen is literally charting the graph that's about to render, not a decoy.
  {
    float t = 6.0f;
    for (int it = 0; it < 600; it++) {
      layout_iter(g, t, -1);
      t = std::max(0.30f, t * 0.99f);
    }
  }
  int N = (int)g.nodes.size();
  std::vector<float> tx(N), ty(N), ox(N), oy(N), appear(N);
  {
    std::mt19937 rng(0xC0FFEEu);
    auto fr = [&](float a, float b) {
      return a + (b - a) * (rng() / float(rng.max()));
    };
    for (int i = 0; i < N; i++) {
      tx[i] = g.nodes[i].x;  // settled target
      ty[i] = g.nodes[i].y;
      float ang = fr(0, 6.2831853f), r = 120 + fr(0, 60);  // scattered far out
      ox[i] = std::cos(ang) * r;                            // fly-in origin
      oy[i] = std::sin(ang) * r;
      appear[i] = fr(0.0f, 0.42f);  // staggered reveal, like stars lighting up
    }
  }
  auto easeOut = [](float t) {  // cubic ease-out: fast in, gentle landing
    t = std::clamp(t, 0.0f, 1.0f);
    float u = 1 - t;
    return 1 - u * u * u;
  };
  auto smooth = [](float t) {  // smoothstep for the UI morph
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3 - 2 * t);
  };

  auto start = std::chrono::steady_clock::now();
  auto secs = [&] {
    return std::chrono::duration<float>(std::chrono::steady_clock::now() - start)
        .count();
  };
  // Intro timeline: nodes fly in and settle (kReveal), hold a beat (kHold),
  // then the app frame assembles around the settled graph (kTrans).
  const float kReveal = 2.4f;
  const float kHold = 0.35f;
  const float kTrans = 0.9f;
  const float kTransStart = kReveal + kHold;
  const float kTransEnd = kTransStart + kTrans;

  enum Phase { INTRO, TRANS, GRAPH };
  Phase phase = INTRO;

  int selected = -1, hover = -1, dragging = -1;
  Box canvas_box;   // filled by reflect() each render
  Camera cam;       // zoom / pan
  int last_GW = 160, last_GH = 100;
  bool panning = false;
  int pan_mx = 0, pan_my = 0;
  std::atomic<bool> paused{false};  // 'space' freezes all motion
  float sim_temp = 0.0f;            // >0 => the layout is actively re-settling

  // Chat-history overlay: a full-screen, scrollable read of one conversation.
  std::atomic<bool> showHist{false};  // read by the ticker to pause repaints
  int histNode = -1;
  int histRow = 0;       // which turn-row is kept in view (drives scrolling)
  int histRowCount = 1;  // set each render, for clamping
  std::unordered_map<int, std::vector<Turn>> histCache;
  auto open_history = [&](int node) {
    if (node < 0)
      return;
    if (!histCache.count(node))
      histCache[node] = load_history(g.nodes[node].txPath);
    histNode = node;
    histRow = 0;
    showHist = true;
  };

  // Shutdown animation: on quit, every node flares and bursts in a wave that
  // travels out from the graph's center, then the app fades and exits.
  std::atomic<bool> shuttingDown{false};
  std::chrono::steady_clock::time_point shutStart;
  const float kShutdown = 1.2f;         // total run time, seconds
  const float kPopDur = 0.22f;          // how long one node's burst lasts
  const float kWaveEnd = 0.66f;         // last node fires by this fraction, so
                                        // the final frames settle to empty
  std::vector<float> popStart(N, 0.0f);  // per-node fire time, in [0,1]
  auto beginShutdown = [&]() {
    if (shuttingDown)
      return;
    showHist = false;
    // Order nodes by distance from the centroid, then hand out fire times evenly
    // by RANK — one node per slice — so the wave still sweeps outward but never
    // clumps, no matter how the graph happens to be spread in space.
    float cxg = 0, cyg = 0;
    for (int i = 0; i < N; i++) { cxg += g.nodes[i].x; cyg += g.nodes[i].y; }
    if (N) { cxg /= N; cyg /= N; }
    std::vector<std::pair<float, int>> byDist(N);
    for (int i = 0; i < N; i++) {
      float dx = g.nodes[i].x - cxg, dy = g.nodes[i].y - cyg;
      byDist[i] = {dx * dx + dy * dy, i};
    }
    std::sort(byDist.begin(), byDist.end());
    std::mt19937 rng(0xBEEFu);
    // Jitter under half a slice: softens the metronome without reordering pops.
    float slice = N > 1 ? kWaveEnd / (N - 1) : 0.0f;
    auto jit = [&] { return (rng() / float(rng.max()) - 0.5f) * slice * 0.8f; };
    for (int r = 0; r < N; r++) {
      float f = (N > 1) ? r / float(N - 1) : 0.0f;  // 0 (center) .. 1 (rim)
      popStart[byDist[r].second] = std::max(0.0f, f * kWaveEnd + jit());
    }
    shutStart = std::chrono::steady_clock::now();
    shuttingDown = true;
  };

  auto neighbors_sorted = [&](int i) {
    std::vector<std::pair<int, float>> v = (i >= 0) ? g.adj[i]
                                                    : std::vector<std::pair<int, float>>{};
    std::sort(v.begin(), v.end(),
              [](auto& a, auto& b) { return a.second > b.second; });
    return v;
  };

  // Mouse cell -> canvas dot coordinates.
  auto to_dot = [&](int mx, int my, float& dx, float& dy) {
    dx = (mx - canvas_box.x_min) * 2.0f;
    dy = (my - canvas_box.y_min) * 4.0f;
  };
  // Canvas dot -> world.
  auto dot_to_world = [&](float dx, float dy, float& wx, float& wy) {
    wx = cam.camx + (dx - last_GW / 2.0f) / cam.zoom;
    wy = cam.camy + (dy - last_GH / 2.0f) / cam.zoom;
  };

  // ------------------------------------------------------------------ render
  // Fade a color up from the near-black backdrop by `a` in [0,1] — used to
  // dissolve the loading chrome out and the app chrome in during the morph.
  auto mix = [](int a, int b, float t) {
    return (uint8_t)std::clamp(a + (b - a) * t, 0.0f, 255.0f);
  };
  auto fade = [&](int r, int gr, int b, float a) {
    return Color::RGB(mix(12, r, a), mix(12, gr, a), mix(14, b, a));
  };

  auto detail_panel = [&]() -> Element {
    if (selected < 0) {
      Elements lines;
      lines.push_back(text("conversation graph") | bold);
      lines.push_back(separator());
      lines.push_back(text(std::to_string(g.nodes.size()) + " conversations"));
      lines.push_back(text(std::to_string(g.edges.size()) + " links"));
      lines.push_back(text(std::to_string(g.nprojects) + " projects"));
      lines.push_back(separator());
      lines.push_back(paragraph("Hover a node to preview it, click to open. "
                                "Bigger nodes had more prompts; color = "
                                "project. Linked nodes share vocabulary.") |
                      dim);
      return vbox(std::move(lines)) | flex;
    }
    auto& v = g.nodes[selected];
    Elements lines;
    lines.push_back(paragraph(v.title) | bold |
                    color(Color::HSV((uint8_t)v.hue, 200, 245)));
    lines.push_back(separator());
    lines.push_back(hbox({text("project ") | dim,
                          text(v.projName.empty() ? "—" : v.projName)}));
    lines.push_back(hbox({text("prompts ") | dim, text(std::to_string(v.prompts))}));
    lines.push_back(hbox({text("first   ") | dim, text(trim_ts(v.tFirst))}));
    lines.push_back(hbox({text("last    ") | dim, text(trim_ts(v.tLast))}));
    lines.push_back(separator());
    lines.push_back(text("first prompt") | dim);
    lines.push_back(paragraph(ellipsis(v.texts.empty() ? "" : v.texts.front(),
                                        160)));
    lines.push_back(separator());
    lines.push_back(
        hbox({text(" ⏎ ") | bold |
                  color(Color::RGB(20, 22, 28)) |
                  bgcolor(Color::HSV((uint8_t)v.hue, 180, 235)),
              text(v.txPath.empty() ? "  no transcript on disk"
                                    : "  open full chat history") |
                  (v.txPath.empty() ? dim : bold)}));
    lines.push_back(separator());
    auto nbrs = neighbors_sorted(selected);
    lines.push_back(text("related  (press 1-6)") | dim);
    if (nbrs.empty())
      lines.push_back(text("  (none)") | dim);
    for (int k = 0; k < (int)nbrs.size() && k < 6; k++) {
      auto& nb = g.nodes[nbrs[k].first];
      lines.push_back(hbox({
          text(" " + std::to_string(k + 1) + " ") | bold,
          text(ellipsis(nb.title, 24)) |
              color(Color::HSV((uint8_t)nb.hue, 180, 235)),
      }));
    }
    return vbox(std::move(lines)) | flex;
  };

  // Full-screen scrollable chat history for the opened conversation.
  auto history_view = [&]() -> Element {
    auto& v = g.nodes[histNode];
    const auto& turns = histCache[histNode];

    Elements rows;
    if (turns.empty()) {
      rows.push_back(text("(no readable messages in this transcript)") | dim);
    }
    for (auto& t : turns) {
      bool you = t.role == 0;
      // Speaker label: "you" in cool blue, "claude" in the project's own hue.
      Color nameCol = you ? Color::RGB(120, 190, 245)
                          : Color::HSV((uint8_t)v.hue, 190, 240);
      rows.push_back(hbox({
          text(you ? "you" : "claude") | bold | color(nameCol),
          text(t.ts.empty() ? "" : "  " + trim_ts(t.ts)) | dim,
      }));
      // Body: one element per line so code blocks and lists keep their shape;
      // over-long lines still wrap via paragraph, tool markers render dim.
      Color bodyCol = you ? Color::RGB(210, 220, 235) : Color::RGB(225, 225, 230);
      std::string ln;
      std::string body = t.text;
      body.push_back('\n');
      for (char c : body) {
        if (c == '\n') {
          if (ln.rfind("· ", 0) == 0)
            rows.push_back(text("  " + ln) | dim | italic);
          else if (ln.empty())
            rows.push_back(text(" "));  // preserve blank lines
          else
            rows.push_back(paragraph(ln) | color(bodyCol));
          ln.clear();
        } else {
          ln.push_back(c);
        }
      }
      rows.push_back(text(" "));  // gap between turns
    }

    histRowCount = std::max(1, (int)rows.size());
    int foc = std::clamp(histRow, 0, histRowCount - 1);
    rows[foc] = rows[foc] | focus;  // yframe scrolls to keep this row in view
    Element doc = vbox(std::move(rows)) | yframe | flex;

    auto header = hbox({
        text(" 💬 ") | bold,
        paragraph(v.title) | bold | color(Color::HSV((uint8_t)v.hue, 200, 245)),
        filler(),
        text(std::to_string((int)turns.size()) + " turns · " +
             (v.projName.empty() ? "—" : v.projName) + " ") |
            dim,
    });
    auto footer = text(" ↑/↓ or j/k scroll · PgUp/PgDn page · g/G top/bottom · "
                       "Esc / ⏎ back to graph · q quit") |
                  dim | hcenter;
    return vbox({
               header,
               separator(),
               doc,
               separator(),
               footer,
           }) |
           border;
  };

  auto renderer = Renderer([&] {
    if (showHist)
      return history_view();

    float el = secs();

    // ---- shutdown: the intro morph in reverse — the panel slides back out and
    // the graph reframes to fill the widening canvas — while every node pops in
    // an outward wave and the chrome fades, then it exits. ----
    if (shuttingDown) {
      float gp = std::chrono::duration<float>(
                     std::chrono::steady_clock::now() - shutStart)
                     .count() /
                 kShutdown;
      if (gp >= 1.0f)
        screen.Exit();  // animation done — leave after this final frame
      std::vector<float> pop(N, 0.0f);
      for (int i = 0; i < N; i++)
        pop[i] = std::clamp((gp - popStart[i]) / kPopDur, 0.0f, 1.0f);

      // m: 1 = full app, 0 = collapsed — the morph, run backwards over the
      // first ~70% so the frame is gone before the last nodes finish popping.
      float m = 1.0f - smooth(std::clamp(gp / 0.7f, 0.0f, 1.0f));

      int budget = screen.dimx() - 4 - (int)std::round(36.0f * m);
      int GW = std::clamp(budget * 2, 80, 480);
      int GH = std::clamp((screen.dimy() - 8) * 4, 80, 360);
      last_GW = GW;
      last_GH = GH;
      camera_fit(g, GW, GH, cam);  // reframe as the panel yields room (stable:
                                   // node positions are frozen during shutdown)
      auto cv = render_graph(g, GW, GH, cam, -1, -1, el, nullptr, &pop);

      char zbuf[32];
      std::snprintf(zbuf, sizeof zbuf, "%.0f%%", cam.zoom / 3.0f * 100.0f);
      auto topbar = hbox({
          text(" ✦ powering down… ") | bold | color(fade(120, 220, 235, m)),
          filler(),
          text(std::to_string(g.nodes.size()) + " · " +
               std::to_string(g.edges.size()) + " links · zoom " + zbuf + " ") |
              color(fade(120, 120, 120, m)),
      });
      auto footer = text(" goodbye ") | color(fade(110, 110, 110, m)) | hcenter;

      int panelW = (int)std::round(34.0f * m);
      Element body =
          panelW <= 0
              ? hbox({canvas(std::move(cv)) | reflect(canvas_box) | flex})
              : hbox({canvas(std::move(cv)) | reflect(canvas_box) | flex,
                      separator(),
                      detail_panel() | size(WIDTH, EQUAL, panelW)});
      return vbox({topbar, separator(), body | flex, separator(), footer}) |
             border;
    }

    // Advance the intro state machine: settle → hold → morph → graph.
    if (phase != GRAPH) {
      if (el >= kTransEnd)
        phase = GRAPH;
      else if (el >= kTransStart)
        phase = TRANS;
    }
    // Morph factor: 0 = loading look, 1 = full app. Only moves during TRANS.
    float tr = (phase == GRAPH)
                   ? 1.0f
                   : smooth((el - kTransStart) / kTrans);

    if (phase != GRAPH) {
      // Fly every node from its scattered origin into its settled target, with
      // a staggered, eased landing. This is the *actual* graph assembling.
      float p = std::clamp(el / kReveal, 0.0f, 1.0f);
      for (int i = 0; i < N; i++) {
        float span = std::max(0.05f, 0.92f - appear[i]);
        float e = easeOut((p - appear[i]) / span);
        g.nodes[i].x = ox[i] + (tx[i] - ox[i]) * e;
        g.nodes[i].y = oy[i] + (ty[i] - oy[i]) * e;
      }
    } else if (!paused && (sim_temp > 0.35f || dragging >= 0)) {
      // GRAPH phase physics: re-settle after a drag/select, or while dragging.
      for (int it = 0; it < 2; it++)
        layout_iter(g, std::max(sim_temp, 0.8f), dragging);
      sim_temp = std::max(0.0f, sim_temp * 0.96f);
    }

    // Per-node reveal alpha for the intro fade-in (all lit once settled).
    std::vector<float> alpha(N, 1.0f);
    if (phase == INTRO) {
      float p = std::clamp(el / kReveal, 0.0f, 1.0f);
      for (int i = 0; i < N; i++)
        alpha[i] = std::clamp((p - appear[i]) / 0.12f, 0.0f, 1.0f);
    }

    // Horizontal budget: the graph canvas yields room as the side panel slides
    // in (panel ≈ 36 cols at tr=1), so the two animate together.
    int budget = screen.dimx() - 4 - (int)std::round(36.0f * tr);
    int GW = std::clamp(budget * 2, 80, 480);
    int GH = std::clamp((screen.dimy() - 8) * 4, 80, 360);
    last_GW = GW;
    last_GH = GH;

    if (phase == GRAPH) {
      if (!cam.user)  // auto-frame until the user takes the camera
        camera_fit(g, GW, GH, cam);
    } else {
      // Frame the *target* layout so the view stays steady while nodes fly in
      // and reframes smoothly as the panel opens.
      float minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
      for (int i = 0; i < N; i++) {
        minx = std::min(minx, tx[i]); maxx = std::max(maxx, tx[i]);
        miny = std::min(miny, ty[i]); maxy = std::max(maxy, ty[i]);
      }
      if (N)
        camera_fit_bounds(minx, maxx, miny, maxy, GW, GH, cam);
    }

    auto cv = render_graph(g, GW, GH, cam, selected, hover, paused ? 0.0f : el,
                           phase == GRAPH ? nullptr : &alpha);

    // ---- top bar: loading title dissolves out, app header dissolves in ----
    Element topbar;
    if (tr < 0.5f) {
      float a = 1.0f - tr / 0.5f;
      topbar = text(" ✦ charting your conversation graph…   any key to skip") |
               bold | color(fade(120, 220, 235, a));
    } else {
      float a = (tr - 0.5f) / 0.5f;
      char zbuf[32];
      std::snprintf(zbuf, sizeof zbuf, "%.0f%%", cam.zoom / 3.0f * 100.0f);
      topbar = hbox({
          text(" claudegraph ") | bold | color(fade(120, 220, 235, a)),
          filler(),
          text(std::to_string(g.nodes.size()) + " · " +
               std::to_string(g.edges.size()) + " links · zoom " + zbuf +
               (paused ? " · paused" : "")) |
              color(fade(120, 120, 120, a)),
      });
    }

    // ---- bottom bar: loading gauge dissolves out, footer dissolves in ----
    Element bottombar;
    if (tr < 0.5f) {
      float a = 1.0f - tr / 0.5f;
      float p = std::clamp(el / kReveal, 0.0f, 1.0f);
      bottombar = hbox({
          text(" loading ") | color(fade(120, 120, 120, a)),
          gauge(p) | color(fade(120, 220, 235, a)) | flex,
          text(" " + std::to_string(int(p * 100)) + "% ") |
              color(fade(120, 120, 120, a)),
      });
    } else {
      float a = (tr - 0.5f) / 0.5f;
      bottombar =
          text(" hover: title · click: select · ⏎: open chat · drag: move · "
               "scroll: zoom · 1-6: related · Tab: cycle · f: fit · space: "
               "freeze · q: quit") |
          color(fade(110, 110, 110, a)) | hcenter;
    }

    // ---- body: side panel slides in from width 0 → 34 as the app opens ----
    int panelW = (int)std::round(34.0f * tr);
    Element body;
    if (panelW <= 0) {
      body = hbox({canvas(std::move(cv)) | reflect(canvas_box) | flex});
    } else {
      body = hbox({
          canvas(std::move(cv)) | reflect(canvas_box) | flex,
          separator(),
          detail_panel() | size(WIDTH, EQUAL, panelW),
      });
    }

    return vbox({
               topbar,
               separator(),
               body | flex,
               separator(),
               bottombar,
           }) |
           border;
  });

  // ------------------------------------------------------------------ input
  auto pick = [&](int mx, int my) -> int {  // mouse cell -> nearest node
    float dotx, doty;
    to_dot(mx, my, dotx, doty);
    int best = -1;
    float bestd = 1e9;
    for (int i = 0; i < (int)g.nodes.size(); i++) {
      float ddx = g.nodes[i].sx - dotx, ddy = g.nodes[i].sy - doty;
      float d = std::sqrt(ddx * ddx + ddy * ddy);
      if (d <= g.nodes[i].sr + 2.5f && d < bestd) {
        bestd = d;
        best = i;
      }
    }
    return best;
  };

  auto ui = CatchEvent(renderer, [&](Event e) {
    // ---- shutting down: swallow input; a second 'q' skips to the exit ----
    if (shuttingDown) {
      if (e == Event::Character('q'))
        screen.Exit();
      return true;
    }

    // ---- chat-history overlay owns all input while it's open ----
    if (showHist) {
      auto step = [&](int d) {
        histRow = std::clamp(histRow + d, 0, histRowCount - 1);
      };
      if (e == Event::Character('q'))
        return beginShutdown(), true;
      if (e == Event::Escape || e == Event::Return ||
          e == Event::Character('o'))
        return showHist = false, true;
      if (e == Event::ArrowDown || e == Event::Character('j'))
        return step(1), true;
      if (e == Event::ArrowUp || e == Event::Character('k'))
        return step(-1), true;
      if (e == Event::PageDown || e == Event::Character(' '))
        return step(12), true;
      if (e == Event::PageUp)
        return step(-12), true;
      if (e == Event::Character('G'))
        return histRow = histRowCount - 1, true;
      if (e == Event::Character('g'))
        return histRow = 0, true;
      if (e.is_mouse()) {
        auto& m = e.mouse();
        if (m.button == Mouse::WheelDown)
          return step(3), true;
        if (m.button == Mouse::WheelUp)
          return step(-3), true;
      }
      return true;  // swallow everything else while reading
    }

    if (phase != GRAPH) {
      if (e.is_character() || e == Event::Return || e == Event::Escape) {
        // Skip straight to the settled graph.
        for (int i = 0; i < N; i++) {
          g.nodes[i].x = tx[i];
          g.nodes[i].y = ty[i];
        }
        start = std::chrono::steady_clock::now() -
                std::chrono::milliseconds(int((kTransEnd + 0.1f) * 1000));
        phase = GRAPH;
        return true;
      }
      return false;
    }

    // GRAPH phase.
    if (e == Event::Character('q'))
      return beginShutdown(), true;  // play the pop-and-fade, then exit
    if (e == Event::Escape)
      return selected = -1, true;
    if (e == Event::Character('f'))
      return cam.user = false, true;  // re-enable auto-fit
    if (e == Event::Character(' '))
      return paused = !paused, true;  // freeze / resume all motion
    if (e == Event::Character('r'))
      return sim_temp = 4.0f, true;   // reheat: let it jostle and re-settle
    if ((e == Event::Return || e == Event::Character('o')) && selected >= 0 &&
        !g.nodes[selected].txPath.empty()) {
      open_history(selected);  // open the full chat for the selected node
      return true;
    }
    if (e == Event::Tab) {
      if (!g.nodes.empty()) {
        selected = (selected + 1) % (int)g.nodes.size();
        sim_temp = std::max(sim_temp, 1.2f);
      }
      return true;
    }
    if (e.is_character() && selected >= 0) {
      std::string ch = e.character();
      if (ch.size() == 1 && ch[0] >= '1' && ch[0] <= '6') {
        auto nbrs = neighbors_sorted(selected);
        int k = ch[0] - '1';
        if (k < (int)nbrs.size()) {
          selected = nbrs[k].first;
          sim_temp = std::max(sim_temp, 1.2f);
          return true;
        }
      }
    }
    if (e.is_mouse()) {
      auto& m = e.mouse();
      // Wheel zoom, anchored on the cursor so it zooms toward the pointer.
      if (m.button == Mouse::WheelUp || m.button == Mouse::WheelDown) {
        float dx, dy, wx, wy;
        to_dot(m.x, m.y, dx, dy);
        dot_to_world(dx, dy, wx, wy);
        float f = (m.button == Mouse::WheelUp) ? 1.18f : 1.0f / 1.18f;
        cam.zoom = std::clamp(cam.zoom * f, 0.4f, 40.0f);
        cam.camx = wx - (dx - last_GW / 2.0f) / cam.zoom;
        cam.camy = wy - (dy - last_GH / 2.0f) / cam.zoom;
        cam.user = true;
        return true;
      }
      if (m.button == Mouse::Left && m.motion == Mouse::Pressed) {
        int hit = pick(m.x, m.y);
        if (hit >= 0) {
          selected = hit;
          dragging = hit;
        } else {  // empty space -> start panning the camera
          panning = true;
          pan_mx = m.x;
          pan_my = m.y;
        }
        return true;
      }
      if (m.motion == Mouse::Moved) {
        if (dragging >= 0) {
          float dx, dy, wx, wy;
          to_dot(m.x, m.y, dx, dy);
          dot_to_world(dx, dy, wx, wy);
          g.nodes[dragging].x = wx;
          g.nodes[dragging].y = wy;
          sim_temp = std::max(sim_temp, 1.5f);  // neighbors follow the drag
        } else if (panning) {
          cam.camx -= (m.x - pan_mx) * 2.0f / cam.zoom;
          cam.camy -= (m.y - pan_my) * 4.0f / cam.zoom;
          pan_mx = m.x;
          pan_my = m.y;
          cam.user = true;
        } else {
          hover = pick(m.x, m.y);
        }
        return true;
      }
      if (m.motion == Mouse::Released) {
        dragging = -1;
        panning = false;
        return true;
      }
    }
    return false;
  });

  // Redraw ticker (~28fps) drives the intro and the graph's gentle idle motion.
  // Pausing ('space') stops the posts, so a frozen graph repaints only on input;
  // the chat overlay is static too, so we don't repaint it between keystrokes.
  std::atomic<bool> running{true};
  std::thread ticker([&] {
    while (running) {
      std::this_thread::sleep_for(std::chrono::milliseconds(36));
      if (shuttingDown || (!paused && !showHist))
        screen.PostEvent(Event::Custom);
    }
  });
  screen.Loop(ui);
  running = false;
  ticker.join();
  return 0;
}
