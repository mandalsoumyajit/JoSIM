// Copyright (c) 2026 JoSIM Developers
// Licensed under the MIT License.
// Standalone JoSIM Co-Simulation IPC Socket Server (josim-cosim-server).
// Supports UNIX domain sockets (--unix /path/to/sock) and TCP sockets (--port <port>).

#include "JoSIM/josim_c_api.h"
#include "JoSIM/josim_cosim_ipc.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
int main() {
    std::cerr << "josim-cosim-server requires POSIX sockets (Linux/WSL/macOS).\n";
    return 1;
}
#else
#include <poll.h>

namespace {

struct ClientContext {
    int                  fd   = -1;
    josim_session_t*     sess = nullptr;
    std::vector<uint8_t> payloadBuf;
    std::vector<double>  outBuf;

    explicit ClientContext(int clientFd, const std::string& defaultNetlistPath) : fd(clientFd) {
        sess = josim_session_create();
        if (!defaultNetlistPath.empty()) {
            josim_load_netlist_file(sess, defaultNetlistPath.c_str());
        }
    }

    ~ClientContext() {
        if (sess) {
            josim_session_destroy(sess);
            sess = nullptr;
        }
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    }

    ClientContext(const ClientContext&)            = delete;
    ClientContext& operator=(const ClientContext&) = delete;
};

int send_response(int fd, int32_t status, const void* payload, uint64_t payload_len) {
    josim_ipc_resp_hdr_t hdr{};
    hdr.magic       = JOSIM_COSIM_MAGIC;
    hdr.status      = status;
    hdr.payload_len = payload_len;
    if (josim_ipc_write_all(fd, &hdr, sizeof(hdr)) != 0) { return -1; }
    if (payload_len > 0 && payload) {
        if (josim_ipc_write_all(fd, payload, static_cast<size_t>(payload_len)) != 0) { return -1; }
    }
    return 0;
}

int send_error(int fd, const char* msg) {
    const std::string s = msg ? msg : "Unknown error";
    return send_response(fd, -1, s.data(), static_cast<uint64_t>(s.size()));
}

void write_finalized_csv(josim_session_t* sess, const std::string& csvSpec) {
    if (csvSpec.empty() || !sess) { return; }
    std::string outCsv = csvSpec;
    std::string evCsv;
    const size_t barPos = csvSpec.find('|');
    if (barPos != std::string::npos) {
        outCsv = csvSpec.substr(0, barPos);
        evCsv  = csvSpec.substr(barPos + 1);
    }

    if (!outCsv.empty()) {
        const int64_t nTr = josim_num_traces(sess);
        const int64_t nSt = josim_num_steps(sess);
        if (nTr > 0 && nSt > 0) {
            std::ofstream out(outCsv);
            if (out.is_open()) {
                std::vector<const double*> ptrs(static_cast<size_t>(nTr), nullptr);
                for (int64_t c = 0; c < nTr; ++c) {
                    if (c > 0) { out << ","; }
                    const char* nm = josim_trace_name(sess, c);
                    out << (nm ? nm : "");
                    ptrs[static_cast<size_t>(c)] = josim_trace_data(sess, c);
                }
                out << "\n";
                out << std::scientific << std::setprecision(10);
                for (int64_t r = 0; r < nSt; ++r) {
                    for (int64_t c = 0; c < nTr; ++c) {
                        if (c > 0) { out << ","; }
                        const double* p = ptrs[static_cast<size_t>(c)];
                        out << (p ? p[r] : 0.0);
                    }
                    out << "\n";
                }
            }
        }

        const int64_t            nEv = josim_num_events(sess);
        const josim_sfq_event_t* evs = josim_events_data(sess);
        std::ofstream            evOut(outCsv + ".events.csv");
        if (evOut.is_open()) {
            evOut << "trace,time,t_peak,v_peak,fwhm,dt_pi,area,phase,polarity,winding,trace_id,trial\n";
            evOut << std::scientific << std::setprecision(12);
            for (int64_t i = 0; i < nEv; ++i) {
                const auto& e  = evs[i];
                const char* tn = josim_event_trace_name(sess, e.trace_id);
                evOut << (tn ? tn : "") << "," << e.time << "," << e.t_peak << "," << e.v_peak << "," << e.fwhm << ","
                      << e.dt_pi << "," << e.area << "," << e.phase << "," << e.polarity << "," << e.winding << ","
                      << e.trace_id << "," << e.trial << "\n";
            }
        }

        int64_t calls = 0, rollbacks = 0, substeps = 0;
        double  ckptSec = 0.0;
        josim_cosim_stats(sess, &calls, &rollbacks, &substeps, &ckptSec);
        std::ofstream metaOut(outCsv + ".meta.txt");
        if (metaOut.is_open()) {
            metaOut << "rollback_count=" << rollbacks << "\n"
                    << "calls=" << calls << "\n"
                    << "rollbacks=" << rollbacks << "\n"
                    << "substeps=" << substeps << "\n"
                    << std::fixed << std::setprecision(6) << "ckpt_time_s=" << ckptSec << "\n";
        }
    }

    if (!evCsv.empty()) {
        const int64_t            nEv = josim_num_events(sess);
        const josim_sfq_event_t* evs = josim_events_data(sess);
        std::vector<size_t>      order(static_cast<size_t>(nEv));
        for (size_t i = 0; i < order.size(); ++i) { order[i] = i; }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return evs[a].time < evs[b].time; });

        std::ofstream evOut(evCsv);
        if (evOut.is_open()) {
            evOut << "trial,trace,time,polarity,winding,t_peak,v_peak,fwhm,dt_pi,area,phase\n";
            evOut << std::scientific << std::setprecision(12);
            for (size_t idx : order) {
                const auto& e  = evs[idx];
                const char* tn = josim_event_trace_name(sess, e.trace_id);
                evOut << e.trial << "," << (tn ? tn : "") << "," << e.time << "," << e.polarity << "," << e.winding
                      << "," << e.t_peak << "," << e.v_peak << "," << e.fwhm << "," << e.dt_pi << "," << e.area << ","
                      << e.phase << "\n";
            }
        }
    }
}

bool handle_client_request(ClientContext& ctx, const std::string& defaultNetlistPath, bool& outShutdownRequested) {
    const int          clientFd   = ctx.fd;
    josim_session_t*&  sess       = ctx.sess;
    auto&              payloadBuf = ctx.payloadBuf;
    auto&              outBuf     = ctx.outBuf;

    josim_ipc_req_hdr_t req{};
    if (josim_ipc_read_all(clientFd, &req, sizeof(req)) != 0) { return false; }
    if (req.magic != JOSIM_COSIM_MAGIC) {
        send_error(clientFd, "Invalid IPC magic header");
        return false;
    }
    payloadBuf.resize(static_cast<size_t>(req.payload_len));
    if (req.payload_len > 0) {
        if (josim_ipc_read_all(clientFd, payloadBuf.data(), static_cast<size_t>(req.payload_len)) != 0) {
            return false;
        }
    }

    if (req.opcode == JOSIM_IPC_OP_SHUTDOWN) {
        send_response(clientFd, 0, nullptr, 0);
        outShutdownRequested = true;
        return false;
    }

    if (req.opcode == JOSIM_IPC_OP_INIT_FILE || req.opcode == JOSIM_IPC_OP_INIT_STRING) {
        if (payloadBuf.size() < 12) {
            send_error(clientFd, "INIT payload too short");
            return true;
        }
        const uint8_t* ptr = payloadBuf.data();
        const uint8_t* end = ptr + payloadBuf.size();

        int32_t  phaseMode  = 0;
        int32_t  nPorts     = 0;
        uint32_t netlistLen = 0;
        std::memcpy(&phaseMode, ptr, 4);
        std::memcpy(&nPorts, ptr + 4, 4);
        std::memcpy(&netlistLen, ptr + 8, 4);
        ptr += 12;

        if (ptr + netlistLen > end) {
            send_error(clientFd, "Truncated netlist in INIT payload");
            return true;
        }
        std::string netlistSpec(reinterpret_cast<const char*>(ptr), netlistLen);
        ptr += netlistLen;

        josim_session_destroy(sess);
        sess = josim_session_create();

        if (phaseMode == 0 || phaseMode == 1) {
            const char* modeStr = (phaseMode == 1) ? "1" : "0";
            const char* argv[2] = {"-a", modeStr};
            josim_set_cli_args(sess, 2, argv);
        }

        int rc = 0;
        if (req.opcode == JOSIM_IPC_OP_INIT_FILE) {
            const std::string& pathToUse = netlistSpec.empty() ? defaultNetlistPath : netlistSpec;
            rc                           = josim_load_netlist_file(sess, pathToUse.c_str());
        } else {
            rc = josim_load_netlist_string(sess, netlistSpec.c_str(), nullptr);
        }
        if (rc != 0) {
            send_error(clientFd, josim_session_last_error(sess));
            return true;
        }

        josim_cosim_clear_ports(sess);
        bool portErr = false;
        for (int32_t p = 0; p < nPorts; ++p) {
            if (ptr + 16 > end) {
                portErr = true;
                break;
            }
            int32_t  mode   = 0;
            double   z0     = 50.0;
            uint16_t posLen = 0;
            uint16_t negLen = 0;
            std::memcpy(&mode, ptr, 4);
            std::memcpy(&z0, ptr + 4, 8);
            std::memcpy(&posLen, ptr + 12, 2);
            std::memcpy(&negLen, ptr + 14, 2);
            ptr += 16;
            if (ptr + posLen + negLen > end) {
                portErr = true;
                break;
            }
            std::string posNode(reinterpret_cast<const char*>(ptr), posLen);
            ptr += posLen;
            std::string negNode(reinterpret_cast<const char*>(ptr), negLen);
            ptr += negLen;
            if (josim_cosim_add_port(sess, posNode.c_str(), negNode.c_str(), mode, z0) < 0) {
                portErr = true;
                break;
            }
        }
        if (portErr) {
            send_error(clientFd, "Invalid port specification in INIT payload");
            return true;
        }

        if (josim_cosim_init(sess) != 0) {
            send_error(clientFd, josim_session_last_error(sess));
            return true;
        }

        struct {
            double  tstep;
            int64_t n_ports;
        } initResp{josim_cosim_tstep(sess), static_cast<int64_t>(josim_cosim_num_ports(sess))};
        send_response(clientFd, 0, &initResp, sizeof(initResp));
    } else if (req.opcode == JOSIM_IPC_OP_STEP_TO) {
        const int32_t nP         = josim_cosim_num_ports(sess);
        const size_t  expectedSz = sizeof(double) * static_cast<size_t>(1 + nP);
        if (payloadBuf.size() < expectedSz) {
            send_error(clientFd, "STEP_TO payload size mismatch");
            return true;
        }
        const double* inDoubles  = reinterpret_cast<const double*>(payloadBuf.data());
        const double  targetTime = inDoubles[0];
        const double* portIn     = (nP > 0) ? (inDoubles + 1) : nullptr;

        outBuf.resize(static_cast<size_t>(2 * nP + 2));
        double* portOut  = outBuf.data();
        double* portDiag = outBuf.data() + nP;
        if (josim_cosim_step_to(sess, targetTime, portIn, portOut, portDiag) != 0) {
            send_error(clientFd, josim_session_last_error(sess));
            return true;
        }
        outBuf[static_cast<size_t>(2 * nP)] = josim_cosim_current_time(sess);
        int64_t rbCnt                       = josim_cosim_rollback_count(sess);
        std::memcpy(&outBuf[static_cast<size_t>(2 * nP + 1)], &rbCnt, sizeof(int64_t));
        send_response(clientFd, 0, outBuf.data(), static_cast<uint64_t>(outBuf.size() * sizeof(double)));
    } else if (req.opcode == JOSIM_IPC_OP_COMMIT) {
        if (josim_cosim_commit(sess) != 0) {
            send_error(clientFd, josim_session_last_error(sess));
            return true;
        }
        struct {
            double  t_cur;
            int64_t rb_cnt;
        } resp{josim_cosim_current_time(sess), josim_cosim_rollback_count(sess)};
        send_response(clientFd, 0, &resp, sizeof(resp));
    } else if (req.opcode == JOSIM_IPC_OP_ROLLBACK) {
        if (josim_cosim_rollback(sess) != 0) {
            send_error(clientFd, josim_session_last_error(sess));
            return true;
        }
        struct {
            double  t_cur;
            int64_t rb_cnt;
        } resp{josim_cosim_current_time(sess), josim_cosim_rollback_count(sess)};
        send_response(clientFd, 0, &resp, sizeof(resp));
    } else if (req.opcode == JOSIM_IPC_OP_RUN_WINDOW) {
        if (payloadBuf.size() < 16) {
            send_error(clientFd, "RUN_WINDOW payload too short");
            return true;
        }
        const int32_t nP     = josim_cosim_num_ports(sess);
        double        tStop  = 0.0;
        int64_t       nInPts = 0;
        std::memcpy(&tStop, payloadBuf.data(), 8);
        std::memcpy(&nInPts, payloadBuf.data() + 8, 8);
        const size_t needBytes = 16 + static_cast<size_t>(nInPts) * static_cast<size_t>(1 + nP) * sizeof(double);
        if (nInPts < 0 || payloadBuf.size() < needBytes) {
            send_error(clientFd, "RUN_WINDOW array payload truncated");
            return true;
        }
        const double* tInPts     = reinterpret_cast<const double*>(payloadBuf.data() + 16);
        const double* portInFlat = tInPts + nInPts;

        int64_t       outNPts  = 0;
        const double* outTPts  = nullptr;
        const double* outYFlat = nullptr;
        if (josim_cosim_run_window(sess, tStop, nInPts, tInPts, portInFlat, &outNPts, &outTPts, &outYFlat) != 0) {
            send_error(clientFd, josim_session_last_error(sess));
            return true;
        }

        const size_t respDoubles = 1 + static_cast<size_t>(outNPts) * static_cast<size_t>(1 + nP);
        outBuf.resize(respDoubles);
        std::memcpy(&outBuf[0], &outNPts, sizeof(int64_t));
        if (outNPts > 0 && outTPts) {
            std::memcpy(&outBuf[1], outTPts, static_cast<size_t>(outNPts) * sizeof(double));
        }
        if (outNPts > 0 && nP > 0 && outYFlat) {
            std::memcpy(&outBuf[1 + outNPts],
                        outYFlat,
                        static_cast<size_t>(outNPts) * static_cast<size_t>(nP) * sizeof(double));
        }
        send_response(clientFd, 0, outBuf.data(), static_cast<uint64_t>(outBuf.size() * sizeof(double)));
    } else if (req.opcode == JOSIM_IPC_OP_FINALIZE) {
        std::string csvPath(reinterpret_cast<const char*>(payloadBuf.data()), payloadBuf.size());
        while (!csvPath.empty() && csvPath.back() == '\0') { csvPath.pop_back(); }

        if (josim_cosim_finalize(sess) != 0) {
            send_error(clientFd, josim_session_last_error(sess));
            return true;
        }
        int64_t calls = 0, rollbacks = 0, substeps = 0;
        double  ckptSec = 0.0;
        josim_cosim_stats(sess, &calls, &rollbacks, &substeps, &ckptSec);
        std::cout << "[josim-cosim-server] calls=" << calls << " rollbacks=" << rollbacks
                  << " substeps=" << substeps << " ckpt_time_s=" << std::fixed << std::setprecision(6) << ckptSec
                  << std::endl;
        if (!csvPath.empty()) {
            write_finalized_csv(sess, csvPath);
        }
        struct {
            int64_t num_traces;
            int64_t num_steps;
            int64_t num_events;
            int64_t rollback_count;
        } resp{josim_num_traces(sess),
               josim_num_steps(sess),
               josim_num_events(sess),
               josim_cosim_rollback_count(sess)};
        send_response(clientFd, 0, &resp, sizeof(resp));
    } else if (req.opcode == JOSIM_IPC_OP_GET_TRACE) {
        std::string name(reinterpret_cast<const char*>(payloadBuf.data()), payloadBuf.size());
        while (!name.empty() && name.back() == '\0') { name.pop_back(); }
        int64_t       len  = 0;
        const double* data = josim_trace_data_by_name(sess, name.c_str(), &len);
        if (!data || len <= 0) {
            send_error(clientFd, ("Trace not found: " + name).c_str());
            return true;
        }
        send_response(clientFd, 0, data, static_cast<uint64_t>(len) * sizeof(double));
    } else if (req.opcode == JOSIM_IPC_OP_GET_EVENTS) {
        const int64_t            nEv = josim_num_events(sess);
        const josim_sfq_event_t* evs = josim_events_data(sess);
        const int64_t            nTr = josim_num_event_traces(sess);

        std::string traceTable;
        for (int32_t i = 0; i < static_cast<int32_t>(nTr); ++i) {
            if (i > 0) { traceTable.push_back('\n'); }
            const char* tn = josim_event_trace_name(sess, i);
            if (tn) { traceTable.append(tn); }
        }
        const uint64_t       tblLen  = static_cast<uint64_t>(traceTable.size());
        const uint64_t       evBytes = static_cast<uint64_t>(nEv) * sizeof(josim_sfq_event_t);
        std::vector<uint8_t> resp(16 + tblLen + evBytes);
        std::memcpy(resp.data(), &nEv, 8);
        std::memcpy(resp.data() + 8, &tblLen, 8);
        if (tblLen > 0) { std::memcpy(resp.data() + 16, traceTable.data(), tblLen); }
        if (evBytes > 0 && evs) { std::memcpy(resp.data() + 16 + tblLen, evs, evBytes); }
        send_response(clientFd, 0, resp.data(), static_cast<uint64_t>(resp.size()));
    } else {
        send_error(clientFd, "Unknown IPC opcode");
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::string unixPath;
    int         tcpPort         = -1;
    std::string netlistPath;
    bool        once            = false;
    int         expectedClients = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: josim-cosim-server [--unix <sock_path> | --port <tcp_port>] "
                         "[--netlist <file.cir>] [--once] [--expected-clients <N>]\n";
            return 0;
        } else if ((arg == "--unix" || arg == "-u") && i + 1 < argc) {
            unixPath = argv[++i];
        } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            tcpPort = std::stoi(argv[++i]);
        } else if ((arg == "--netlist" || arg == "-n") && i + 1 < argc) {
            netlistPath = argv[++i];
        } else if (arg == "--once") {
            once = true;
        } else if (arg == "--expected-clients" && i + 1 < argc) {
            expectedClients = std::max(1, std::stoi(argv[++i]));
            once            = true;
        }
    }

    if (unixPath.empty() && tcpPort < 0) {
        std::cerr << "Usage: josim-cosim-server [--unix <sock_path> | --port <tcp_port>] "
                     "[--netlist <file.cir>] [--once] [--expected-clients <N>]\n";
        return 1;
    }

    int listenFd = -1;
    if (!unixPath.empty()) {
        unlink(unixPath.c_str());
        listenFd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (listenFd < 0) {
            std::perror("socket(AF_UNIX)");
            return 1;
        }
        struct sockaddr_un sun{};
        sun.sun_family = AF_UNIX;
        std::strncpy(sun.sun_path, unixPath.c_str(), sizeof(sun.sun_path) - 1);
        if (bind(listenFd, reinterpret_cast<struct sockaddr*>(&sun), sizeof(sun)) != 0) {
            std::perror("bind(AF_UNIX)");
            close(listenFd);
            return 1;
        }
        if (listen(listenFd, 32) != 0) {
            std::perror("listen(AF_UNIX)");
            close(listenFd);
            return 1;
        }
        std::cout << "JOSIM_COSIM_SERVER_READY unix:" << unixPath << std::endl;
    } else {
        listenFd = socket(AF_INET, SOCK_STREAM, 0);
        if (listenFd < 0) {
            std::perror("socket(AF_INET)");
            return 1;
        }
        int one = 1;
        setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        struct sockaddr_in sin{};
        sin.sin_family      = AF_INET;
        sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        sin.sin_port        = htons(static_cast<uint16_t>(tcpPort));
        if (bind(listenFd, reinterpret_cast<struct sockaddr*>(&sin), sizeof(sin)) != 0) {
            std::perror("bind(AF_INET)");
            close(listenFd);
            return 1;
        }
        if (listen(listenFd, 32) != 0) {
            std::perror("listen(AF_INET)");
            close(listenFd);
            return 1;
        }
        socklen_t len = sizeof(sin);
        getsockname(listenFd, reinterpret_cast<struct sockaddr*>(&sin), &len);
        const int boundPort = ntohs(sin.sin_port);
        std::cout << "JOSIM_COSIM_SERVER_READY tcp:127.0.0.1:" << boundPort << std::endl;
    }

    // Multiplexed poll() event loop supporting concurrent multi-partition SPICE/Spectre sessions
    std::vector<std::unique_ptr<ClientContext>> clients;
    std::vector<struct pollfd>                  pfds;
    int                                         totalAccepted = 0;
    bool                                        shutdownReq   = false;

    while (!shutdownReq) {
        pfds.clear();
        pfds.push_back(pollfd{listenFd, POLLIN, 0});
        for (const auto& c : clients) {
            pfds.push_back(pollfd{c->fd, POLLIN, 0});
        }

        const int nReady = poll(pfds.data(), static_cast<nfds_t>(pfds.size()), -1);
        if (nReady < 0) {
            if (errno == EINTR) { continue; }
            break;
        }

        // 1. Check existing clients for incoming requests or disconnections
        for (size_t i = clients.size(); i > 0; --i) {
            const size_t idx = i - 1;
            const short  rev = pfds[idx + 1].revents;
            if (rev & (POLLIN | POLLHUP | POLLERR)) {
                if (!handle_client_request(*clients[idx], netlistPath, shutdownReq)) {
                    clients.erase(clients.begin() + static_cast<std::ptrdiff_t>(idx));
                }
            }
        }

        // 2. Check listening socket for new client connections
        if (pfds[0].revents & POLLIN) {
            int clientFd = accept(listenFd, nullptr, nullptr);
            if (clientFd >= 0) {
                if (unixPath.empty()) {
                    int one = 1;
                    setsockopt(clientFd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
                }
                clients.push_back(std::make_unique<ClientContext>(clientFd, netlistPath));
                ++totalAccepted;
            }
        }

        if (once && totalAccepted >= expectedClients && clients.empty()) {
            break;
        }
    }

    clients.clear();
    close(listenFd);
    if (!unixPath.empty()) { unlink(unixPath.c_str()); }
    return 0;
}
#endif
