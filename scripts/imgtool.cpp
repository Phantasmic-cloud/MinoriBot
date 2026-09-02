/*
从文件读入二进制形式的图片数据，进行图片操作
参数1: 文件名:str 参数2: 输出文件名:str 参数3: 容差:int
输入文件格式: n:int, h:int, w:int, r00:uint8_t g00:uint8_t b00:uint8_t, a00:uint8_t, ...
*/

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <cassert> 
#include <cstring>

constexpr int dx[4] = {0, 0, -1, 1};
constexpr int dy[4] = {-1, 1, 0, 0};

union Color {
    struct {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };
    uint32_t key;
    Color() = default;
    Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a) : r(r), g(g), b(b), a(a) {}
    Color(uint32_t key) : key(key) {}
};

int n, h, w;
Color *img = nullptr;

Color& get_color(int t, int y, int x) {
    return img[t * h * w + y * w + x];
}
int quad(int x) {
    return x * x;
}
int color_diff(const Color& a, const Color& b) {
    return quad(int(a.r) - int(b.r)) + quad(int(a.g) - int(b.g)) + quad(int(a.b) - int(b.b));
}
bool check_pos(int y, int x) {
    return y >= 0 && y < h && x >= 0 && x < w;
}

void floodfill(int t, int sy, int sx, const Color& src, const Color& dst, int tolerance) {
    assert(!dst.a);
    static std::vector<std::tuple<int, int>> stack;
    stack.clear();
    stack.emplace_back(sy, sx);
    get_color(t, sy, sx) = dst;

    // std::cerr << "[cutout] start floodfill (" << t << ", " << sy << ", " << sx << ")" << std::endl;

    while (!stack.empty()) {
        auto [y, x] = stack.back();
        stack.pop_back();

        // std::cerr << "[cutout] floodfill (" << t << ", " << y << ", " << x << ")" << std::endl;
        
        for (int i = 0; i < 4; ++i) {
            int ny = y + dy[i];
            int nx = x + dx[i];
            if (!check_pos(ny, nx)) continue;
            auto& c = get_color(t, ny, nx);
            if (!c.a) continue;
            if (color_diff(c, src) > tolerance) continue;
            c = dst;
            stack.emplace_back(ny, nx);
        }
    }
}


int main(int argc, char *argv[]) {
    std::string filename = argv[1];
    std::string outname = argv[2];
    std::string command = argv[3];
    
    // 读取图片数据
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        std::cerr << "[imgtool-cpp] error opening file: " << filename << std::endl;
        return 1;
    }
    fread(&n, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    fread(&w, sizeof(int), 1, fp);
    long long size = n * h * w * sizeof(Color);
    if (size > 1e9) {
        std::cerr << "[imgtool-cpp] error: image size too large" << std::endl;
        fclose(fp);
        return 1;
    }
    img = new Color[n * h * w];
    auto read = fread(img, sizeof(Color), n * h * w, fp);
    assert(read == n * h * w);
    fclose(fp);

    // 返回的额外数据
    std::string extra_ret{};

    // cutout
    if (command == "cutout") {
        int tolerance = atoi(argv[4]);
        tolerance = (tolerance * tolerance) * 3;

        std::cerr << "[imgtool-cpp] start to cutout img (" << n << "x" << h << "x" << w << ")" << std::endl;
        // 以第一帧的边缘像素作为参照，找出最常见的颜色
        std::unordered_map<uint32_t, int> edge_color_count{};
        Color max_color;
        int max_color_count = 0;

        auto update_color = [&](Color c) {
            int cnt = ++edge_color_count[c.key];
            if (cnt > max_color_count) {
                max_color_count = cnt;
                max_color = c;
            }
        };

        for (int y = 0; y < h; ++y) { 
            update_color(get_color(0, y, 0));
            update_color(get_color(0, y, w - 1));
        }
        for (int x = 0; x < w; ++x) {
            update_color(get_color(0, 0, x));
            update_color(get_color(0, h - 1, x));
        }

        std::cerr << "[imgtool-cpp] max color: " << int(max_color.r) << " " << int(max_color.g) << " " << int(max_color.b) << " " << int(max_color.a) << std::endl;

        // 遍历每一帧，从边缘开始抠图
        for (int t = 0; t < n; ++t) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    if (x == 0 || x == w - 1 || y == 0 || y == h - 1) {
                        auto& c = get_color(t, y, x);
                        if (c.a && color_diff(c, max_color) <= tolerance) 
                            floodfill(t, y, x, max_color, Color(0, 0, 0, 0), tolerance);
                    }
                }
            }
        }

        std::cerr << "[imgtool-cpp] cutout done" << std::endl;
    }
    // shrink
    else if (command == "shrink") {
        int alpha_threshold = atoi(argv[4]);
        int edge = atoi(argv[5]);
        // 计算alpha>threshold的最小包围盒
        int x0 = w;
        int y0 = h;
        int x1 = 0;
        int y1 = 0;
        for (int t = 0; t < n; ++t) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    auto& c = get_color(t, y, x);
                    if (c.a > alpha_threshold) {
                        if (x < x0) x0 = x;
                        if (y < y0) y0 = y;
                        if (x > x1) x1 = x;
                        if (y > y1) y1 = y;
                    }
                }
            }
        }
        
        int nw = x1 - x0 + 1;
        int nh = y1 - y0 + 1;
        Color* new_img = new Color[n * (nh + 2 * edge) * (nw + 2 * edge)];
        memset(new_img, 0, n * (nh + 2 * edge) * (nw + 2 * edge) * sizeof(Color));
        
        for (int t = 0; t < n; ++t) {
            for (int y = 0; y < nh + 2 * edge; ++y) {
                for (int x = 0; x < nw + 2 * edge; ++x) {
                    int src_x = x0 + x - edge;
                    int src_y = y0 + y - edge;
                    if (src_x >= 0 && src_x < w && src_y >= 0 && src_y < h) {
                        new_img[t * (nh + 2 * edge) * (nw + 2 * edge) + y * (nw + 2 * edge) + x] = get_color(t, src_y, src_x);
                    }
                }
            }
        }
        h = nh + 2 * edge;
        w = nw + 2 * edge;
        delete[] img;
        img = new_img;

        // 返回json格式的bbox，返回图像对应原图坐标
        int bx = x0 - edge;
        int by = y0 - edge;
        int bw = nw + 2 * edge;
        int bh = nh + 2 * edge;
        extra_ret = "{\"bbox\":[" + std::to_string(bx) + "," + std::to_string(by) + "," + std::to_string(bw) + "," + std::to_string(bh) + "]}";
    }
    else {
        throw std::runtime_error("[imgtool-cpp] unknown command: " + command);
    }

    // 输出处理完毕的图像
    FILE *out_fp = fopen(outname.c_str(), "wb");
    if (!out_fp) {
        std::cerr << "[imgtool-cpp] error opening output file: " << outname << std::endl;
        delete[] img;
        return 1;
    }
    fwrite(&n, sizeof(int), 1, out_fp);
    fwrite(&h, sizeof(int), 1, out_fp);
    fwrite(&w, sizeof(int), 1, out_fp);
    fwrite(img, sizeof(Color), n * h * w, out_fp);

    // 输出额外数据
    int extra_ret_len = extra_ret.size();
    fwrite(&extra_ret_len, sizeof(int), 1, out_fp);
    if (extra_ret_len > 0) 
        fwrite(extra_ret.data(), sizeof(char), extra_ret_len, out_fp);

    fclose(out_fp);
    delete[] img;
    return 0;
}