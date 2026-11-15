#include "fcntl.h"
#include "poll.h"
#include "signal.h"
#include "supervision/process.hpp"
#include "sys/prctl.h"
#include "sys/types.h"
#include "sys/wait.h"
#include "time.h"
#include "unistd.h"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace specferry::supervision {
namespace {
volatile sig_atomic_t interrupted = 0;

void on_signal(int number) { interrupted = number; }

double clock_seconds(clockid_t clock) {
  timespec now{};
  if (clock_gettime(clock, &now)) {
    throw std::runtime_error("cannot read supervisor clock");
  }
  return now.tv_sec + now.tv_nsec * 1e-9;
}

double now() { return clock_seconds(CLOCK_MONOTONIC); }

class Descriptor {
public:
  explicit Descriptor(int fd = -1) : fd_(fd) {}

  ~Descriptor() { close(); }

  Descriptor(const Descriptor &) = delete;
  Descriptor &operator=(const Descriptor &) = delete;

  int get() const { return fd_; }

  void close() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

private:
  int fd_;
};

void nonblocking(int fd) {
  const int flags = fcntl(fd, F_GETFL);
  if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK)) {
    throw std::runtime_error("cannot make supervisor pipe nonblocking");
  }
}

// JSON output only: request parsing is owned by the existing deployment adapter.
std::string json_string(const std::string &text) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char value : text) {
    if (value == '"' || value == '\\') {
      out << '\\' << value;
    } else if (value < 32) {
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(value) << std::dec;
    } else {
      out << value;
    }
  }
  out << '"';
  return out.str();
}

void publish(const std::filesystem::path &path, const std::string &text) {
  auto temporary = path;
  temporary += ".tmp";
  std::ofstream output(temporary);
  output << text << '\n';
  output.close();
  if (!output) {
    throw std::runtime_error("cannot publish supervisor evidence: " + path.string());
  }
  std::filesystem::rename(temporary, path);
}

std::vector<int> group_members(pid_t group) {
  std::vector<int> members;
  for (const auto &entry : std::filesystem::directory_iterator("/proc")) {
    const auto name = entry.path().filename().string();
    if (name.empty() || name.find_first_not_of("0123456789") != std::string::npos) {
      continue;
    }
    std::ifstream file(entry.path() / "stat");
    std::string stat;
    if (!std::getline(file, stat)) {
      continue;
    }
    const auto end = stat.rfind(')');
    if (end == std::string::npos) {
      throw std::runtime_error("invalid process stat record");
    }
    std::istringstream fields(stat.substr(end + 1));
    char state;
    int parent = 0, process_group = 0;
    if (!(fields >> state >> parent >> process_group)) {
      throw std::runtime_error("invalid process group record");
    }
    if (process_group == group && state != 'Z') {
      members.push_back(std::stoi(name));
    }
  }
  return members;
}

struct SignalAction {
  int signal;
  double when;
};

struct Child {
  pid_t pid = -1;
  bool exited = false;
  int status = 0;
  std::vector<SignalAction> actions;

  void poll() {
    if (pid <= 0 || exited) {
      return;
    }
    const auto result = waitpid(pid, &status, WNOHANG);
    if (result == pid) {
      exited = true;
    } else if (result < 0 && errno != EINTR) {
      throw std::runtime_error("cannot reap supervised child");
    }
  }

  void terminate() {
    if (pid <= 0) {
      return;
    }
    // A leader can exit while descendants still own resources. Always signal its group.
    for (const int request : {SIGTERM, SIGKILL}) {
      if (kill(-pid, request)) {
        if (errno == ESRCH) {
          // The child can still be between fork and setsid. Signal the known PID too.
          if (!exited && kill(pid, request) == 0) {
            actions.push_back({request, now()});
          }
        } else {
          throw std::runtime_error("cannot signal supervised process group");
        }
      } else {
        actions.push_back({request, now()});
      }
      if (request == SIGTERM) {
        kill(-pid, SIGCONT);
      }
      const auto until = now() + 3;
      do {
        poll();
        if (exited) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      } while (now() < until);
    }
  }

  ~Child() {
    try {
      terminate();
    } catch (...) {
      // Explicit termination records failures; an existing exception must survive cleanup.
    }
  }
};

struct Evidence {
  double started = now();
  double wall = clock_seconds(CLOCK_REALTIME);
  std::optional<double> ready;
  std::string reason;
  std::string error;
  std::vector<int> remaining;
  bool child_started = false;
  std::optional<int> exec_error;
  double elapsed = 0;
};

std::string evidence_json(const ProcessOptions &options, const Child &child, const Evidence &record,
                          bool complete) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"child_start_monotonic_seconds\":" << record.started
      << ",\"child_start_wall_seconds\":" << record.wall
      << ",\"child_started\":" << (record.child_started ? "true" : "false")
      << ",\"supervisor_implementation\":\"cpp\",\"stop_reason\":"
      << (record.reason.empty() ? "null" : json_string(record.reason));
  if (child.pid > 0) {
    out << ",\"child_pid\":" << child.pid;
  }
  if (options.preparation_seconds) {
    out << ",\"preparation_limit_seconds\":" << *options.preparation_seconds
        << ",\"inference_limit_seconds\":" << options.execution_seconds;
  }
  if (record.ready) {
    out << ",\"ready_monotonic_seconds\":" << *record.ready
        << ",\"preparation_seconds\":" << *record.ready - record.started;
  }
  if (!record.error.empty()) {
    out << ",\"error\":" << json_string(record.error);
  }
  if (record.exec_error) {
    out << ",\"spawn_error\":" << json_string(std::strerror(*record.exec_error));
  }
  out << ",\"actions\":[";
  for (std::size_t i = 0; i < child.actions.size(); ++i) {
    const auto &action = child.actions[i];
    out << (i ? "," : "") << "{\"signal\":\"" << (action.signal == SIGTERM ? "SIGTERM" : "SIGKILL")
        << "\",\"monotonic_seconds\":" << action.when << '}';
  }
  out << "],\"remaining_processes\":[";
  for (std::size_t i = 0; i < record.remaining.size(); ++i) {
    out << (i ? "," : "") << record.remaining[i];
  }
  out << ']';
  if (complete) {
    out << ",\"exit\":";
    if (!child.exited) {
      out << "null";
    } else if (WIFSIGNALED(child.status)) {
      out << -WTERMSIG(child.status);
    } else {
      out << WEXITSTATUS(child.status);
    }
    out << ",\"child_seconds\":" << record.elapsed;
    if (record.ready) {
      out << ",\"post_ready_child_seconds\":" << record.started + record.elapsed - *record.ready;
    }
  }
  return out.str() + '}';
}

void validate(const ProcessOptions &options) {
  if ((options.execution_limit != 300 && options.execution_limit != 600) ||
      (options.execution_limit == 600 && !options.preparation_seconds) ||
      !std::isfinite(options.execution_seconds) || options.execution_seconds <= 0 ||
      options.execution_seconds > options.execution_limit || options.command.empty()) {
    throw std::invalid_argument("invalid supervisor execution budget or command");
  }
  if (options.preparation_seconds &&
      (!std::isfinite(*options.preparation_seconds) || *options.preparation_seconds <= 0 ||
       *options.preparation_seconds > 240)) {
    throw std::invalid_argument("invalid supervisor preparation budget");
  }
}
} // namespace

int supervise(const ProcessOptions &options) {
  validate(options);
  interrupted = 0;

  struct sigaction handler {};

  handler.sa_handler = on_signal;
  sigemptyset(&handler.sa_mask);
  if (sigaction(SIGTERM, &handler, nullptr) || sigaction(SIGINT, &handler, nullptr)) {
    throw std::runtime_error("cannot install supervisor signal handlers");
  }
  signal(SIGPIPE, SIG_IGN);
  const auto owner = getppid();
  if (prctl(PR_SET_PDEATHSIG, SIGTERM) || getppid() != owner || owner == 1) {
    throw std::runtime_error("supervisor policy owner disappeared");
  }
  nonblocking(STDIN_FILENO);
  int ready_pipe[2], exec_pipe[2];
  if (pipe2(ready_pipe, O_CLOEXEC)) {
    throw std::runtime_error("cannot create readiness pipe");
  }
  Descriptor ready_read(ready_pipe[0]), ready_write(ready_pipe[1]);
  if (pipe2(exec_pipe, O_CLOEXEC)) {
    throw std::runtime_error("cannot create exec status pipe");
  }
  Descriptor exec_read(exec_pipe[0]), exec_write(exec_pipe[1]);
  nonblocking(ready_read.get());
  nonblocking(exec_read.get());
  Descriptor log(open((options.output / "native.log").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600));
  if (log.get() < 0) {
    throw std::runtime_error("cannot open model log");
  }
  std::vector<char *> command;
  for (const auto &part : options.command) {
    command.push_back(const_cast<char *>(part.c_str()));
  }
  command.push_back(nullptr);
  Child child;
  Evidence record;
  const auto guard_pid = getpid();
  child.pid = fork();
  if (child.pid < 0) {
    throw std::runtime_error("cannot fork model process");
  }
  if (child.pid == 0) {
    const auto fail = [&] {
      const auto code = errno;
      const auto ignored = write(exec_write.get(), &code, sizeof(code));
      (void)ignored;
      _exit(127);
    };
    if (setsid() < 0 || prctl(PR_SET_PDEATHSIG, SIGKILL) || getppid() != guard_pid) {
      fail();
    }
    Descriptor input(open("/dev/null", O_RDONLY));
    if (input.get() < 0 || dup2(input.get(), STDIN_FILENO) < 0 ||
        dup2(log.get(), STDOUT_FILENO) < 0 || dup2(log.get(), STDERR_FILENO) < 0 ||
        chdir(options.working_directory.c_str())) {
      fail();
    }
    if (options.preparation_seconds) {
      if (fcntl(ready_write.get(), F_SETFD, 0) ||
          setenv("SPECFERRY_READY_FD", std::to_string(ready_write.get()).c_str(), 1)) {
        fail();
      }
    } else {
      unsetenv("SPECFERRY_READY_FD");
    }
    execvp(command[0], command.data());
    fail();
  }
  ready_write.close();
  exec_write.close();
  record.child_started = true;
  std::string phase = options.preparation_seconds ? "preparation" : "child";
  double deadline =
      record.started + options.preparation_seconds.value_or(options.execution_seconds);
  std::string ready_text, control_text;
  const auto observe_ready = [&] {
    if (!options.preparation_seconds || record.ready) {
      return;
    }
    char buffer[128];
    const auto count = read(ready_read.get(), buffer, sizeof(buffer));
    if (count < 0 && errno != EAGAIN && errno != EINTR) {
      throw std::runtime_error("cannot read readiness pipe");
    }
    if (count > 0) {
      ready_text.append(buffer, count);
    }
    if (ready_text.size() > 64) {
      throw std::runtime_error("invalid model readiness notification: oversized record");
    }
    if (ready_text.find('\n') == std::string::npos) {
      return;
    }
    std::istringstream fields(ready_text);
    double ready;
    std::string extra;
    if (!(fields >> ready) || fields >> extra || !std::isfinite(ready) || ready < record.started ||
        ready > now() || ready > record.started + *options.preparation_seconds) {
      throw std::runtime_error("invalid model readiness notification: invalid timestamp");
    }
    record.ready = ready;
    deadline = ready + options.execution_seconds;
    phase = "inference";
  };
  try {
    // The policy process publishes this small record. A blocked SD write there cannot
    // delay this watchdog. The pipe is empty and the write is bounded/nonblocking.
    nonblocking(STDOUT_FILENO);
    const auto initial = evidence_json(options, child, record, false) + '\n';
    if (write(STDOUT_FILENO, initial.data(), initial.size()) !=
        static_cast<ssize_t>(initial.size())) {
      throw std::runtime_error("cannot publish child identity to policy owner");
    }
    // Do not publish files in this loop: SD IO must not delay deadline enforcement.
    while (true) {
      child.poll();
      observe_ready();
      if (child.exited) {
        break;
      }
      if (interrupted) {
        record.reason = "supervisor interrupted by signal " + std::to_string(interrupted);
        break;
      }
      if (now() >= deadline) {
        record.reason = phase + " deadline expired";
        break;
      }
      char buffer[1024];
      const auto count = read(STDIN_FILENO, buffer, sizeof(buffer));
      if (count == 0) {
        record.reason = "policy owner closed stop channel";
        break;
      }
      if (count < 0 && errno != EAGAIN && errno != EINTR) {
        throw std::runtime_error("cannot read supervisor stop channel");
      }
      if (count > 0) {
        control_text.append(buffer, count);
        if (control_text.size() > 4096) {
          record.reason = "invalid policy stop notification";
          break;
        }
        const auto end = control_text.find('\n');
        if (end != std::string::npos) {
          record.reason = control_text.substr(0, end);
          if (record.reason.empty()) {
            record.reason = "policy requested stop";
          }
          break;
        }
      }
      pollfd control{STDIN_FILENO, POLLIN | POLLHUP, 0};
      if (::poll(&control, 1, 20) < 0 && errno != EINTR) {
        throw std::runtime_error("supervisor poll failed");
      }
    }
  } catch (const std::exception &error) {
    record.reason = error.what();
    record.error = error.what();
  }
  const auto stopped_at = now();
  try {
    child.terminate();
    int code = 0;
    if (read(exec_read.get(), &code, sizeof(code)) == sizeof(code)) {
      record.exec_error = code;
      record.child_started = false;
      record.reason = "model exec failed";
    }
    const auto drain_deadline = now() + 3;
    do {
      record.remaining = group_members(child.pid);
      if (record.remaining.empty()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (now() < drain_deadline);
  } catch (const std::exception &error) {
    record.error = error.what();
  }
  if (options.preparation_seconds && !record.ready && record.reason.empty() && child.exited &&
      child.status == 0) {
    record.reason = "model readiness not reported";
  }
  record.elapsed = now() - record.started;
  if (!record.reason.empty()) {
    std::ostringstream watchdog;
    watchdog << std::setprecision(17) << "{\"expired_at\":" << stopped_at
             << ",\"child_pid\":" << child.pid << ",\"phase\":" << json_string(phase) << '}';
    publish(options.output / "watchdog.json", watchdog.str());
  }
  publish(options.output / "child.json", evidence_json(options, child, record, true));
  // Do not repeat group signalling after recording the final resource snapshot.
  child.pid = -1;
  return 0;
}
} // namespace specferry::supervision
