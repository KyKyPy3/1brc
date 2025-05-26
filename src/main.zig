const std = @import("std");
const tracy = @import("tracy");

// Пул строк для path
const StringPool = struct {
    allocator: std.mem.Allocator,
    hash_map: std.StringHashMap(void),

    fn init(allocator: std.mem.Allocator) StringPool {
        return .{
            .allocator = allocator,
            .hash_map = std.StringHashMap(void).init(allocator),
        };
    }

    fn deinit(self: *StringPool) void {
        self.hash_map.deinit();
    }

    fn intern(self: *StringPool, str: []const u8) ![]const u8 {
        if (self.hash_map.getKey(str)) |key| {
            if (std.mem.eql(u8, key, str)) {
                return key;
            }
        }

        const duped = try self.allocator.dupe(u8, str);
        try self.hash_map.put(duped, {});

        return duped;
    }
};

// Статистика для каждого эндпоинта
const EndpointStats = struct {
    min_response_time: i32,
    max_response_time: i32,
    total_response_time: i64,
    count: u32,

    fn update(self: *EndpointStats, response_time: u64) void {
        self.min_response_time = @min(self.min_response_time, response_time);
        self.max_response_time = @max(self.max_response_time, response_time);
        self.total_response_time += response_time;
        self.count += 1;
    }

    fn getAvgResponseTime(self: EndpointStats) f64 {
        if (self.count == 0) return 0;
        return @as(f64, @floatFromInt(self.total_response_time)) / @as(f64, @floatFromInt(self.count));
    }
};

pub const Record = struct {
    path: []const u8,
    count: u32 = 0,
    total: i64 = 0,
    min: i32 = std.math.maxInt(i32),
    max: i32 = std.math.minInt(i32),

    pub fn init(path: []const u8, time: i32) !Record {
        return Record{
            .path = path,
            .count = 1,
            .total = time,
            .min = time,
            .max = time,
        };
    }

    pub fn update(self: *Record, time: i32) void {
        self.count += 1;
        self.total += time;
        if (time < self.min) self.min = time;
        if (time > self.max) self.max = time;
    }
};

const Map = std.StringHashMap(Record);

var threadMaps: []Map = undefined;

pub fn main() !void {
    const zone = tracy.zone(@src());
    defer zone.end();

    const start = std.time.nanoTimestamp();
    const stdout = std.io.getStdOut().writer();

    // Get command line args
    const args = try std.process.argsAlloc(std.heap.page_allocator);
    defer std.process.argsFree(std.heap.page_allocator, args);

    const filename = args[1];

    const cpu_num = try std.Thread.getCpuCount();

    const parts = try splitFile(std.heap.page_allocator, filename, cpu_num);
    defer parts.deinit();

    var pool: std.Thread.Pool = undefined;
    try pool.init(std.Thread.Pool.Options{ .allocator = std.heap.page_allocator, .n_jobs = cpu_num });
    defer pool.deinit();

    threadMaps = try std.heap.page_allocator.alloc(Map, cpu_num);
    for (threadMaps) |*map| {
        map.* = Map.init(std.heap.page_allocator);

        // Pre-allocate HashMaps with capacity to reduce rehashing
        try map.ensureTotalCapacity(8192);
    }

    var wait_group = std.Thread.WaitGroup{};

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    for (parts.items, 0..) |part, i| {
        wait_group.start();
        try pool.spawn(work, .{ alloc, filename, i, part.offset, part.length, &wait_group });
    }

    pool.waitAndWork(&wait_group);

    var global_stats = std.StringHashMap(EndpointStats).init(std.heap.page_allocator);
    defer global_stats.deinit();

    for (threadMaps) |threadMap| {
        var it = threadMap.iterator();
        while (it.next()) |entry| {
            const stats = entry.value_ptr.*;

            const gop = try global_stats.getOrPut(stats.path);
            if (!gop.found_existing) {
                gop.value_ptr.* = EndpointStats{
                    .min_response_time = stats.min,
                    .max_response_time = stats.max,
                    .total_response_time = stats.total,
                    .count = stats.count,
                };
            } else {
                // Объединяем статистику
                gop.value_ptr.min_response_time = @min(gop.value_ptr.min_response_time, stats.min);
                gop.value_ptr.max_response_time = @max(gop.value_ptr.max_response_time, stats.max);
                gop.value_ptr.total_response_time += stats.total;
                gop.value_ptr.count += stats.count;
            }
        }
    }

    // Сортируем ключи
    var keys = std.ArrayList([]const u8).init(std.heap.page_allocator);
    defer keys.deinit();

    var key_it = global_stats.keyIterator();
    while (key_it.next()) |key| {
        try keys.append(key.*);
    }

    // Сортируем по алфавиту
    std.mem.sort([]const u8, keys.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);

    // Выводим результаты в JSON формате
    try stdout.writeAll("{\n  \"endpoints\": {\n");

    for (keys.items, 0..) |key, i| {
        const stats = global_stats.get(key).?;

        try stdout.print("    \"{s}\": {{\n", .{key});
        try stdout.print("      \"min_response_time\": {d},\n", .{stats.min_response_time});
        try stdout.print("      \"avg_response_time\": {d:.1},\n", .{stats.getAvgResponseTime()});
        try stdout.print("      \"max_response_time\": {d}\n", .{stats.max_response_time});

        if (i == keys.items.len - 1) {
            try stdout.writeAll("    }\n");
        } else {
            try stdout.writeAll("    },\n");
        }
    }

    try stdout.writeAll("  }\n}\n");

    const end = std.time.nanoTimestamp();
    const seconds: f64 = @as(f64, @floatFromInt(end - start)) / 1_000_000_000;
    try stdout.print("Process data took {d} seconds\n", .{seconds});
}

const Part = struct {
    offset: u64,
    length: u64,
};

fn splitFile(allocator: std.mem.Allocator, filename: []const u8, partsNum: usize) !std.ArrayList(Part) {
    const zone = tracy.zone(@src());
    defer zone.end();

    var parts = try std.ArrayList(Part).initCapacity(allocator, partsNum);

    const cwd = std.fs.cwd();
    var file = try cwd.openFile(filename, .{});
    defer file.close();

    const fileInfo = try cwd.statFile(filename);
    const fileSize = fileInfo.size;

    const chunkSize = fileSize / partsNum;

    if (chunkSize < 4096 or partsNum == 1) {
        try parts.append(.{ .offset = 0, .length = fileSize });
        return parts;
    }

    var buf: [1024]u8 = undefined;
    var offset: u64 = 0;
    var i: u8 = 0;

    while (i < partsNum) : (i = i + 1) {
        if (i == partsNum - 1) {
            if (offset < fileSize) {
                try parts.append(.{ .offset = offset, .length = fileSize - offset });
            }
            break;
        }

        const target_offset = offset + chunkSize;

        if (target_offset >= fileSize) {
            try parts.append(.{ .offset = offset, .length = fileSize - offset });
            break;
        }

        const seekOffset = @max(target_offset - buf.len, 0);
        try file.seekTo(seekOffset);
        const n = try file.readAll(&buf);

        const newLine = std.mem.lastIndexOf(u8, buf[0..n], "\n");
        if (newLine) |nl| {
            const nextOffset = seekOffset + nl + 1;
            try parts.append(.{ .offset = offset, .length = nextOffset - offset });
            offset = nextOffset;
        } else {
            try parts.append(.{ .offset = offset, .length = target_offset - offset });
            offset = target_offset;
        }
    }

    return parts;
}

fn work(allocator: std.mem.Allocator, filename: []const u8, threadId: usize, offset: u64, partSize: u64, wait_group: *std.Thread.WaitGroup) void {
    const zone = tracy.zone(@src());
    defer zone.end();

    defer wait_group.finish();

    const cwd = std.fs.cwd();
    var file = cwd.openFile(filename, .{ .mode = .read_only }) catch {
        std.debug.print("Can't open file", .{});

        return;
    };
    defer file.close();

    file.seekTo(offset) catch {
        std.debug.print("Can't seek file", .{});

        return;
    };

    var string_pool = StringPool.init(allocator);
    defer string_pool.deinit();

    const threadMap = &threadMaps[threadId];

    // Создаем буффер для чтения данных из файла
    // Создаем его в куче, потому что 32Мб не влезет в стек треда
    const chunkSize: usize = 32 * 1024 * 1024;
    var buf = std.heap.page_allocator.alloc(u8, chunkSize) catch {
        std.debug.print("Failed to allocate reading buffer\n", .{});
        return;
    };
    defer std.heap.page_allocator.free(buf);
    var bytesRead: usize = 0;

    var remainder = std.ArrayList(u8).initCapacity(std.heap.page_allocator, 4096) catch {
        std.debug.print("Failed to allocate remainder buffer\n", .{});
        return;
    };
    defer remainder.deinit();

    var lineBuff = std.ArrayList(u8).initCapacity(std.heap.page_allocator, 4096) catch {
        std.debug.print("Failed to allocate line buffer\n", .{});
        return;
    };
    defer lineBuff.deinit();

    while (bytesRead < partSize) {
        const read_file_zone = tracy.zoneEx(@src(), .{ .name = "Read file chunk" });
        defer read_file_zone.end();

        const bytesToRead: usize = @min(chunkSize, partSize - bytesRead);
        const n = file.read(buf[0..bytesToRead]) catch {
            std.debug.print("Can't read from file\n", .{});
            break;
        };

        if (n == 0) {
            break;
        }

        bytesRead += n;

        const lastNewLine = std.mem.lastIndexOf(u8, buf[0..n], "\n");
        if (lastNewLine) |ln| {
            if (remainder.items.len > 0) {
                lineBuff.appendSlice(remainder.items) catch {
                    std.debug.print("Failed to append slice to line buffer\n", .{});
                };
                lineBuff.appendSlice(buf[0 .. ln + 1]) catch {
                    std.debug.print("Failed to append slice to line buffer\n", .{});
                };

                processLines(lineBuff.items, threadMap, &string_pool);

                lineBuff.clearRetainingCapacity();
                remainder.clearRetainingCapacity();
            } else {
                processLines(buf[0 .. ln + 1], threadMap, &string_pool);
            }

            if (ln < n - 1) {
                remainder.clearRetainingCapacity();
                remainder.insertSlice(0, buf[ln + 1 .. n]) catch {};
            }
        } else {
            remainder.appendSlice(buf[0..n]) catch {
                std.debug.print("Failed to append slice to remainder\n", .{});
            };
        }
    }

    // Если что-то осталось необработанным
    if (remainder.items.len > 0) {
        processLines(remainder.items, threadMap, &string_pool);
    }
}

fn processLines(lines: []const u8, threadMap: *Map, string_pool: *StringPool) void {
    var spaceCount: u8 = 0;
    var pathStart: usize = 0;
    var pathEnd: usize = 0;
    var timeStart: usize = 0;

    const zone = tracy.zone(@src());
    defer zone.end();

    var i: usize = 0;
    while (i < lines.len) : (i += 1) {
        const c = lines[i];
        if (c == ' ') {
            spaceCount += 1;

            switch (spaceCount) {
                3 => pathStart = i + 1,
                4 => pathEnd = i,
                5 => timeStart = i + 1,
                else => {},
            }

            continue;
        }

        if (c == '\n') {
            const process_results_zone = tracy.zoneEx(@src(), .{ .name = "Process results" });
            const path = string_pool.intern(lines[pathStart..pathEnd]) catch {
                std.debug.print("Failed to intern path: {s}\n", .{lines[pathStart..pathEnd]});
                return;
            };
            const responseTime = std.fmt.parseInt(i32, lines[timeStart..i], 10) catch {
                std.debug.print("Failed to parse response time: {s}\n", .{lines[timeStart..i]});
                return;
            };

            if (threadMap.getPtr(path)) |record| {
                record.update(responseTime);
            } else {
                if (Record.init(path, responseTime)) |record| {
                    threadMap.put(record.path, record) catch {
                        std.debug.print("Failed to put record into thread map\n", .{});
                    };
                } else |_| {}
            }

            process_results_zone.end();
            spaceCount = 0;
        }
    }
}
