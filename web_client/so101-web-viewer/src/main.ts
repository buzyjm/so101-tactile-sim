/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

import './index.css';
import { AppStreamer, DirectConfig, StreamEvent, StreamProps, LogLevel, StreamType, EventAction, EventStatus } from '@nvidia/ov-web-rtc';

interface AppState {
    streamFailed: boolean;
    errorMessage: string;
}

class StreamingApp {
    private streamConnected = false;
    private streamRequested = false;
    private state: AppState = {
        streamFailed: false,
        errorMessage: 'FAILED TO CONNECT TO STREAM',
    };

    private updateUI() {
        const mainDiv = document.getElementById('main-div');
        const errorMessage = document.getElementById('error-message');
        const loadingMessage = document.getElementById('loading-stream-message');
        if (!mainDiv || !errorMessage || !loadingMessage) return;
        const setDisplay = (el, val) => { if (el) el.style.display = val; };
        if (this.state.streamFailed) {
            setDisplay(mainDiv, 'none');
            setDisplay(errorMessage, 'flex');
            setDisplay(loadingMessage, 'none');
            const errorText = document.getElementById('error-message-text');
            if (errorText) errorText.textContent = this.state.errorMessage;
        } else if (this.streamConnected) {
            setDisplay(mainDiv, 'block');
            setDisplay(errorMessage, 'none');
            setDisplay(loadingMessage, 'none');
        } else {
            setDisplay(mainDiv, 'none');
            setDisplay(errorMessage, 'none');
            setDisplay(loadingMessage, 'flex');
        }
        }


    public async initialize() {
        this.updateUI();
        await this.initializeStream();
    }

    private async initializeStream() {
        if (this.streamConnected || this.streamRequested) {
            return;
        }
        this.streamRequested = true;
        const query = new URLSearchParams(window.location.search);
        const signalingServer = query.get('server') || window.location.hostname;
        const streamConfig: DirectConfig = {
            videoElementId: 'remote-video',
            // Use the same host that served this page. This works both for
            // localhost and for a trusted LAN client opening the server IP.
            // The query override handles IDE/browser proxies whose hostname is
            // not the machine running Isaac Sim.
            signalingServer,
            signalingPort: 49100,
            width: 1920,
            height: 1080,
            fps: 60,
            onStart: (message: StreamEvent) => {
                if (message.action === EventAction.START) {
                    if (message.status === EventStatus.SUCCESS) {
                        this.streamConnected = true;
                        console.debug('Stream Ready');
                        this.updateUI();
                    }
                    else if (message.status === EventStatus.ERROR) {
                        console.error('Stream start error:', message.info);
                        this.state.streamFailed = true;
                        this.state.errorMessage = `${message.info || 'Unknown error'}`;
                        this.updateUI();
                    }
                }
            },
            onCustomEvent: (message: any) => {
                console.log('Custom event:', message);
            },
            onStop: (message: StreamEvent) => {
                this.streamConnected = false;
                console.log('Stream stopped:', message);
                this.updateUI();
            },
        };
        const streamProps: StreamProps = {
            streamSource: StreamType.DIRECT,
            logLevel: LogLevel.INFO,
            streamConfig,
        };
        try {
            const result = await AppStreamer.connect(streamProps);
            console.info(`Success: ${result.info}`);
        }
        catch (error) {
            console.error('Failed to connect to stream:', error);
            this.streamRequested = false;
            this.state.streamFailed = true;
            this.state.errorMessage = `${error || 'Unknown error'}`;
            this.updateUI();
        }
    }
}

// Initialize the application
const app = new StreamingApp();
void app.initialize();
