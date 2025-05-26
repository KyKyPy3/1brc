const std = @import("std");

const TracyStep = struct {
    step: std.Build.Step,

    fn create(b: *std.Build) *TracyStep {
        const self = b.allocator.create(TracyStep) catch unreachable;
        self.* = .{
            .step = std.Build.Step.init(.{
                .id = .custom,
                .name = "open-tracy",
                .owner = b,
                .makeFn = make,
            }),
        };
        return self;
    }

    fn make(step: *std.Build.Step, _: std.Build.Step.MakeOptions) !void {
        var tracy_proc = std.process.Child.init(&.{ "tracy-profiler", "-a", "127.0.0.1" }, step.owner.allocator);
        tracy_proc.stdin_behavior = .Ignore;
        tracy_proc.stdout_behavior = .Inherit;
        tracy_proc.stderr_behavior = .Inherit;
        try tracy_proc.spawn();
    }
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const ztracy_dep = b.dependency("tracy", .{
        .target = target,
        .optimize = optimize,
        .enable_tracing = !(b.option(bool, "disable-tracing", "") orelse false),
    });

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe_mod.addImport("tracy", ztracy_dep.module("tracy"));

    const exe = b.addExecutable(.{
        .name = "tracy",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const tracy_step = TracyStep.create(b);
    tracy_step.step.dependOn(b.getInstallStep());

    const tracy_build_step = b.step("tracy", "Run the app then open the tracy gui from the PATH `tracy-profiler`");
    tracy_build_step.dependOn(&tracy_step.step);
    tracy_build_step.dependOn(&run_cmd.step);

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
